import torch
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import Dict, List, Union, Optional, Tuple, Iterator, Any


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def convert_ids_to_string(
        tokenizer: PreTrainedTokenizer,
        ids: torch.LongTensor) -> str:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer.convert_tokens_to_string(tokens)


def experimental_clip_gradient_norm_(
        gradients: List[torch.Tensor],
        max_norm: Optional[float] = 1.0) -> None:

    if max_norm is None:
        return gradients

    total_norm = torch.norm(torch.stack(
        [torch.norm(grad, 2) for grad in gradients]), 2)
    clip_coef = max_norm / (total_norm + 1e-6)

    for grad in gradients:
        grad.detach().mul_(clip_coef)


def get_loss_with_weight_decay(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]]) -> float:

    # model.train()
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    outputs = model(**inputs)
    # model outputs are always tuple in transformers (see doc)
    loss = outputs[0]

    if n_gpu > 1:
        # mean() to average on multi-gpu parallel training
        loss = loss.mean()

    # In PyTorch, weight-decay loss and gradients are calculated in
    # optimizers rather in nn.Module, so we have to manually specify
    # this for the loss here.
    if weight_decay is not None:
        no_decay = (
            weight_decay_ignores
            if weight_decay_ignores
            is not None else [])

        weight_decay_loss = torch.cat([
            p.square().view(-1)
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]).sum() * weight_decay
        loss = loss + weight_decay_loss

    return loss


def compute_gradients(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]]
) -> List[torch.FloatTensor]:

    if params_filter is None:
        params_filter = []

    model.zero_grad()
    loss = get_loss_with_weight_decay(
        device=device, n_gpu=n_gpu,
        model=model, inputs=inputs,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores)

    return torch.autograd.grad(
        outputs=loss,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        create_graph=True)


def compute_hessian_vector_products(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        vectors: torch.FloatTensor,
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]]
) -> List[torch.FloatTensor]:

    if params_filter is None:
        params_filter = []

    model.zero_grad()
    loss = get_loss_with_weight_decay(
        model=model, n_gpu=n_gpu,
        device=device, inputs=inputs,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores)

    grad_tuple = torch.autograd.grad(
        outputs=loss,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        create_graph=True)

    model.zero_grad()
    grad_grad_tuple = torch.autograd.grad(
        outputs=grad_tuple,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        grad_outputs=vectors,
        only_inputs=True
    )

    return grad_grad_tuple


def compute_s_test(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        train_data_loaders: List[torch.utils.data.DataLoader],
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        damp: float,
        scale: float,
        num_samples: Optional[int] = None,
        verbose: bool = True,
) -> List[torch.FloatTensor]:

    v = compute_gradients(
        model=model,
        n_gpu=n_gpu,
        device=device,
        inputs=test_inputs,
        params_filter=params_filter,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores)

    # Technically, it's hv^-1
    last_estimate = list(v).copy()
    cumulative_num_samples = 0
    with tqdm(total=num_samples) as pbar:
        for data_loader in train_data_loaders:
            for i, inputs in enumerate(data_loader):
                this_estimate = compute_hessian_vector_products(
                    model=model,
                    n_gpu=n_gpu,
                    device=device,
                    vectors=last_estimate,
                    inputs=inputs,
                    params_filter=params_filter,
                    weight_decay=weight_decay,
                    weight_decay_ignores=weight_decay_ignores)
                # Recursively caclulate h_estimate
                # https://github.com/dedeswim/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_functions/hvp_grad.py#L118
                with torch.no_grad():
                    new_estimate = [
                        a + (1 - damp) * b - c / scale
                        for a, b, c in zip(v, last_estimate, this_estimate)
                    ]

                pbar.update(1)
                if verbose is True:
                    new_estimate_norm = new_estimate[0].norm().item()
                    last_estimate_norm = last_estimate[0].norm().item()
                    estimate_norm_diff = new_estimate_norm - last_estimate_norm
                    pbar.set_description(f"{new_estimate_norm:.2f} | {estimate_norm_diff:.2f}")

                cumulative_num_samples += 1
                last_estimate = new_estimate
                if num_samples is not None and i > num_samples:
                    break

    # References:
    # https://github.com/kohpangwei/influence-release/blob/master/influence/genericNeuralNet.py#L475
    # Do this for each iteration of estimation
    # Since we use one estimation, we put this at the end
    inverse_hvp = [X / scale for X in last_estimate]
    # Do this at the end
    inverse_hvp = [X / cumulative_num_samples for X in inverse_hvp]
    return inverse_hvp


def compute_grad_zs(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
) -> List[List[torch.FloatTensor]]:

    if weight_decay_ignores is None:
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]

    grad_zs = []
    for inputs in data_loader:
        grad_z = compute_gradients(
            n_gpu=n_gpu, device=device,
            model=model, inputs=inputs,
            params_filter=params_filter,
            weight_decay=weight_decay,
            weight_decay_ignores=weight_decay_ignores)
        with torch.no_grad():
            grad_zs.append([X.cpu() for X in grad_z])

    return grad_zs


def compute_influences(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        batch_train_data_loader: torch.utils.data.DataLoader,
        instance_train_data_loader: torch.utils.data.DataLoader,
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
        s_test_iterations: int = 1,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        train_indices_to_include: Optional[Union[np.ndarray, List[int]]] = None,
) -> Tuple[Dict[int, float], Dict[int, Dict], List[torch.FloatTensor]]:

    if s_test_iterations < 1:
        raise ValueError("`s_test_iterations` must >= 1")

    if weight_decay_ignores is None:
        # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]

    if precomputed_s_test is not None:
        s_test = precomputed_s_test
    else:
        s_test = None
        for _ in range(s_test_iterations):
            _s_test = compute_s_test(
                n_gpu=n_gpu,
                device=device,
                model=model,
                test_inputs=test_inputs,
                train_data_loaders=[batch_train_data_loader],
                params_filter=params_filter,
                weight_decay=weight_decay,
                weight_decay_ignores=weight_decay_ignores,
                damp=s_test_damp,
                scale=s_test_scale,
                num_samples=s_test_num_samples)

            # Sum the values across runs
            if s_test is None:
                s_test = _s_test
            else:
                s_test = [
                    a + b for a, b in zip(s_test, _s_test)
                ]
        # Do the averaging
        s_test = [a / s_test_iterations for a in s_test]

    influences = {}
    train_inputs_collections = {}
    for index, train_inputs in enumerate(tqdm(instance_train_data_loader)):

        # Skip indices when a subset is specified to be included
        if (train_indices_to_include is not None) and (
                index not in train_indices_to_include):
            continue

        grad_z = compute_gradients(
            n_gpu=n_gpu,
            device=device,
            model=model,
            inputs=train_inputs,
            params_filter=params_filter,
            weight_decay=weight_decay,
            weight_decay_ignores=weight_decay_ignores)

        with torch.no_grad():
            influence = [
                - torch.sum(x * y)
                for x, y in zip(grad_z, s_test)]

        influences[index] = sum(influence).item()
        train_inputs_collections[index] = train_inputs

    return influences, train_inputs_collections, s_test
