import os
import sys
import torch
import tempfile
import numpy as np
import torch.distributed as dist
# import torch.multiprocessing as mp
from tqdm import tqdm
from copy import deepcopy
from transformers import GlueDataset
from typing import Dict, List, Any, Tuple, Union, Optional

from experiments import misc_utils
from influence_utils import nn_influence_utils
from influence_utils import multiprocessing_utils as custom_mp


def _compute_s_test(
        rank: int,
        model: torch.nn.Module,
        dataloaders: torch.utils.data.DataLoader,
        n_gpu: int,
        devices: List[torch.device],
        test_inputs: Dict[str, torch.Tensor],
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
) -> List[torch.Tensor]:

    s_test = nn_influence_utils.compute_s_test(
        n_gpu=n_gpu,
        device=devices[rank],
        model=model,
        test_inputs=test_inputs,
        train_data_loaders=[dataloaders],
        params_filter=params_filter,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores,
        damp=s_test_damp,
        scale=s_test_scale,
        num_samples=s_test_num_samples)

    # Gather `s_test` computed in other processes and
    # aggregate them via averaging.
    # print(flatten_and_concat(s_test).norm())
    world_size = float(dist.get_world_size())
    for index in range(len(s_test)):
        dist.all_reduce(s_test[index], op=dist.ReduceOp.SUM)
        s_test[index] = s_test[index] / world_size
    # print(flatten_and_concat(s_test).norm())
    return s_test


def _compute_influences(
        rank: int,
        model: torch.nn.Module,
        s_test: List[torch.Tensor],
        scattered_inputs: List[Any],
        scattered_indices: List[int],
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
) -> Dict[int, float]:

    wrapped_model = InfluenceHelper(
        mode="list",
        n_gpu=1,
        model=model,
        progress_bar=True,
        params_filter=params_filter,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores)

    influences_list = wrapped_model(
        Xs=scattered_inputs,
        s_test=s_test)

    influences = {}
    for i, index in enumerate(scattered_indices):
        # Save just the values not the Tensor to
        # speed up saving/loading time
        influences[index] = influences_list[i].item()

    return influences


def compute_s_test_and_influence(
        rank: int,
        file_name: str,
        model: torch.nn.Module,
        dataloaders: torch.utils.data.DataLoader,
        scattered_inputs: List[Any],
        scattered_indices: List[int],
        n_gpu: int,
        devices: List[torch.device],
        test_inputs: Dict[str, torch.Tensor],
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
        return_s_test: bool = False,
        log_stdin_and_stdout: bool = True,
) -> Tuple[Dict[int, float], List[torch.FloatTensor]]:

    # Initialize
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=len(devices))

    if log_stdin_and_stdout is True:
        logdir = "./logs"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        # https://stackoverflow.com/questions/1501651/log-output-of-multiprocessing-process
        sys.stdout = open(os.path.join(logdir, f"mp.{os.getpid()}.out"), "a")
        sys.stderr = open(os.path.join(logdir, f"mp.{os.getpid()}.err"), "a")

    # Approx. 4-5sec for moving model to a specified GPU
    model.to(devices[rank])
    s_test = _compute_s_test(
        rank=rank,
        model=model,
        dataloaders=dataloaders,
        n_gpu=n_gpu,
        devices=devices,
        test_inputs=test_inputs,
        params_filter=params_filter,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores,
        s_test_damp=s_test_damp,
        s_test_scale=s_test_scale,
        s_test_num_samples=s_test_num_samples)

    influences = _compute_influences(
        rank=rank,
        model=model,
        s_test=s_test,
        scattered_inputs=scattered_inputs,
        scattered_indices=scattered_indices,
        params_filter=params_filter,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores)

    if log_stdin_and_stdout is True:
        # https://stackoverflow.com/questions/14245227/python-reset-stdout-to-normal-after-previously-redirecting-it-to-a-file
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # Save outputs, normally we do not need
    # need `s_test` and saving it takes extra time,
    # but sometimes we need it for diagnostics.
    if return_s_test is True:
        torch.save({
            "influences": influences,
            "s_test": s_test},
            file_name)

    else:
        torch.save({
            "influences": influences},
            file_name)

    # Always return both, though in multiprocessing
    # this does not matter except making type annotation
    # cleaner, which is also important :)
    return influences, s_test


def compute_influences_parallel(
        device_ids: List[int],
        train_dataset: GlueDataset,
        batch_size: int,
        model: torch.nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
        random: bool = True,
        debug: bool = False,
        return_s_test: bool = False,
        train_indices_to_include: Optional[Union[np.ndarray, List[int]]] = None,
) -> Tuple[Dict[int, float], Optional[List[torch.FloatTensor]]]:

    if s_test_num_samples is None:
        raise ValueError("`s_test_num_samples` cannot be None")

    # Passing the smaller subset of training data to child
    # processes can significantly reduce the overhead of
    # spawning new child processes.
    dataloders = prepare_small_dataloaders(
        dataset=train_dataset,
        random=random,
        batch_size=batch_size,
        num_datasets=len(device_ids),
        num_examples_per_dataset=s_test_num_samples)

    scattered_inputs, scattered_indices = prepare_scattered_inputs_and_indices(
        dataset=train_dataset,
        device_ids=device_ids,
        indices_to_include=train_indices_to_include)

    devices = [torch.device(f"cuda:{device_id}") for device_id in device_ids]
    tmpfiles = [tempfile.NamedTemporaryFile() for _ in range(len(device_ids))]
    process_args = [(
        tmpfiles[process_index].name,
        model,
        dataloders[process_index],
        scattered_inputs[process_index],
        scattered_indices[process_index],
        1,  # n_gpu
        devices,
        test_inputs,
        params_filter,
        weight_decay,
        weight_decay_ignores,
        s_test_damp,
        s_test_scale,
        s_test_num_samples,
        return_s_test,
        True if debug is False else False,  # log_stdin_and_stdout
    ) for process_index in range(len(device_ids))]

    if debug is False:
        try:
            custom_mp.spawn(
                compute_s_test_and_influence,
                list_of_args=process_args,
                nprocs=len(device_ids),
                join=True)

            influences: Dict[int, float] = {}
            for tmpfile in tmpfiles:
                outputs_dict = torch.load(tmpfile.name)
                for key, val in outputs_dict["influences"].items():
                    if key in influences.keys():
                        raise ValueError
                    influences[key] = val

            # Note that `s_test` is the same across all processes
            # after they end because of the syncronization, so we
            # just need to load one of them, and we pick the last one
            s_test = outputs_dict.get("s_test", None)

        finally:
            for tmpfile in tmpfiles:
                tmpfile.close()

        return influences, s_test

    else:
        random_rank = np.random.choice(len(device_ids))
        print(f"Using random rank {random_rank}")
        return compute_s_test_and_influence(
            random_rank, *process_args[random_rank])


class SimpleDataset(torch.utils.data.Dataset):
    """Simple Dataset class where examples are fetched by index"""

    def __init__(self, examples: List[Any]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Any:
        return self.examples[index]


def prepare_small_dataloaders(
        dataset: torch.utils.data.Dataset,
        random: bool,
        batch_size: int,
        num_datasets: int,
        num_examples_per_dataset: int) -> List[SimpleDataset]:
    """Only pass to child processes the data we will really use"""

    examples = []
    total_num_examples = batch_size * num_datasets * num_examples_per_dataset

    if random is True:
        indices = np.random.choice(
            len(dataset),
            size=total_num_examples,
            # Sample without replacement
            replace=False)
    else:
        indices = list(range(total_num_examples))

    for index in indices:
        example = dataset[index]
        examples.append(example)

    dataloaders = []
    for i in range(num_datasets):
        start_index = i * batch_size * num_examples_per_dataset
        end_index = (i + 1) * batch_size * num_examples_per_dataset
        new_dataset = SimpleDataset(examples[start_index: end_index])
        dataloader = misc_utils.get_dataloader(
            dataset=new_dataset,
            batch_size=batch_size,
            # The random here doesn't matter?
            random=random)
        dataloaders.append(dataloader)

    return dataloaders


def prepare_scattered_inputs_and_indices(
        device_ids: List[int],
        dataset: torch.utils.data.Dataset,
        indices_to_include: Optional[List[int]] = None,
) -> Tuple[List[List[Any]], List[List[int]]]:
    """Scatter the data into devices"""

    indices_list = []
    # inputs_collections = {}
    inputs_collections_list = []
    instance_dataloader = misc_utils.get_dataloader(
        dataset=dataset, batch_size=1)
    for index, train_inputs in enumerate(tqdm(instance_dataloader)):

        # Skip indices when a subset is specified to be included
        if (indices_to_include is not None) and (
                index not in indices_to_include):
            continue

        indices_list.append(index)
        # inputs_collections[index] = train_inputs
        inputs_collections_list.append(train_inputs)

    scattered_inputs, scattered_indices = scatter_inputs_and_indices(
        Xs=inputs_collections_list,
        indices=indices_list, device_ids=device_ids)

    return scattered_inputs, scattered_indices


def flatten_and_concat(Xs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([X.flatten() for X in Xs], dim=0)


def scatter_inputs_and_indices(
        Xs: List[Any],
        indices: List[int],
        device_ids: List[int]
) -> Tuple[List[List[Any]], List[List[int]]]:
    """Scatter `Xs` across devices"""
    copied_Xs = deepcopy(Xs)
    copied_indices = deepcopy(indices)
    devices = [torch.device(f"cuda:{i}") for i in device_ids]

    def _map_to_device(X: Any, device: torch.device):
        for k, v in X.items():
            if isinstance(v, torch.Tensor):
                X[k] = v.to(device)

        return X

    scattered_Xs: List[List[Any]] = [[] for _ in range(len(device_ids))]
    scattered_indices: List[List[int]] = [[] for _ in range(len(device_ids))]
    boundary = np.ceil(len(copied_Xs) / len(device_ids))
    for i, (X, index) in enumerate(zip(copied_Xs, copied_indices)):
        device_index = int(i // boundary)
        device = devices[device_index]
        scattered_Xs[device_index].append(
            _map_to_device(X, device))
        scattered_indices[device_index].append(index)

    return scattered_Xs, scattered_indices


class InfluenceHelper(torch.nn.Module):
    """Helper Module for computing influence values"""

    def __init__(self,
                 mode: str,
                 n_gpu: int,
                 model: torch.nn.Module,
                 progress_bar: bool = False,
                 params_filter: Optional[List[str]] = None,
                 weight_decay: Optional[float] = None,
                 weight_decay_ignores: Optional[List[str]] = None):

        super(InfluenceHelper, self).__init__()

        if mode not in ["list", "instance"]:
            raise ValueError

        if weight_decay_ignores is None:
            # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
            weight_decay_ignores = [
                "bias",
                "LayerNorm.weight"]

        self.model = model
        self._mode = mode
        self._n_gpu = n_gpu
        self._progress_bar = progress_bar
        self._params_filter = params_filter
        self._weight_decay = weight_decay
        self._weight_decay_ignores = weight_decay_ignores

    def _compute_influence(
            self,
            device: torch.device,
            X: Dict[str, torch.Tensor],
            s_test: List[torch.FloatTensor],
    ) -> torch.Tensor:

        grad_z = nn_influence_utils.compute_gradients(
            n_gpu=self._n_gpu,
            device=device,
            model=self.model,
            inputs=X,
            params_filter=self._params_filter,
            weight_decay=self._weight_decay,
            weight_decay_ignores=self._weight_decay_ignores)

        with torch.no_grad():
            influence = [
                - torch.sum(x * y)
                for x, y in zip(grad_z, s_test)]

        return sum(influence)

    def forward(self,
                Xs: Union[Dict[str, torch.Tensor],
                          List[Dict[str, torch.Tensor]]],
                s_test: List[torch.FloatTensor]
                ) -> torch.FloatTensor:

        if self._mode in ["instance"]:
            # `Xs` has single instance
            if not isinstance(Xs, dict):
                raise TypeError(f"`Xs` should be a dictionary but {type(Xs)}")

            device = Xs["labels"].device
            new_s_test = [x.to(device) for x in s_test]
            return self._compute_influence(
                device=device, X=Xs, s_test=new_s_test)

        else:
            # `Xs` is a list of instances
            if not isinstance(Xs, list):
                raise TypeError(f"`Xs` should be a list but {type(Xs)}")

            influences = []
            device = Xs[0]["labels"].device
            new_s_test = [x.to(device) for x in s_test]
            if self._progress_bar is True:
                Xs = tqdm(Xs)

            influences = [
                self._compute_influence(
                    device=device, X=X,  # noqa
                    s_test=new_s_test)
                for X in Xs]

            return torch.stack(influences, dim=0)
