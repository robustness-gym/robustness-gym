import torch
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Union
import torch.nn.functional as F
import torch.autograd.functional as AF


def compute_batch_Hessian(
        device: torch.device,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        weight_decay: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

    hessian: Optional[List[List[torch.FloatTensor]]] = None
    total_num_data_points = 0
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        def _forward(W: torch.FloatTensor,
                     b: torch.FloatTensor) -> torch.FloatTensor:
            logits = F.linear(data.view(-1, 28 * 28), W, b)
            if model._binary is True:
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(dim=-1), target.float())
            else:
                loss = F.cross_entropy(logits, target)

            loss += weight_decay * (W.square().sum() + b.square().sum())
            return loss

        num_data_points = data.shape[0]
        total_num_data_points += num_data_points
        weight, bias = [p[1] for p in model.named_parameters()]

        # Compute batch Hessian
        _hessian = AF.hessian(func=_forward, inputs=(weight, bias))

        # Cumulate Hessians
        if hessian is None:
            # Make them mutable lists rather than tuples
            hessian = [[x * num_data_points for x in _h]
                       for _h in _hessian]
        else:
            # H: [m, m]
            for i in range(len(hessian)):
                for j in range(len(hessian[0])):
                    hessian[i][j] += _hessian[i][j] * num_data_points

    if hessian is None:
        raise ValueError("`hessian` is None")

    # Transform [
    #     [ |w| x |w|, |w| x |b| ],
    #     [ |b| x |w|, |b| x |b| ]
    # ]
    # into [ |w| + |b|, |w| + |b| ]
    w_size = model.linear.out_features * model.linear.in_features
    b_size = model.linear.out_features
    H = torch.cat([
        # [ |w|, |w| + |b| ]
        torch.cat([x.view(w_size, -1) for x in hessian[0]], dim=-1),
        # [ |b|, |w| + |b| ]
        torch.cat([x.view(b_size, -1) for x in hessian[1]], dim=-1)
    ], dim=0)

    H = H / total_num_data_points
    # H is not invertible here, is pinv good enough?
    return H, H.pinverse()


def compute_instance_Jacobians(
        device: torch.device,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        weight_decay: float,
) -> List[torch.FloatTensor]:

    jacobians = []
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)
        # We need Jacobian per instance
        for instance_idx in range(data.shape[0]):
            _data = data[[instance_idx], ...]
            _target = target[[instance_idx], ...]

            def _forward(W: torch.FloatTensor,
                         b: torch.FloatTensor) -> torch.FloatTensor:

                # Use instance level `_data` and `_target`
                # instead of batch level `data` and `target`
                logits = F.linear(_data.view(-1, 28 * 28), W, b)
                if model._binary is True:
                    loss = F.binary_cross_entropy_with_logits(
                        logits.squeeze(dim=-1), _target.float())
                else:
                    loss = F.cross_entropy(logits, _target)

                loss += weight_decay * (W.square().sum() + b.square().sum())
                return loss

            weight, bias = [p[1] for p in model.named_parameters()]

            # Compute batch Jacobian
            _jacobian = AF.jacobian(func=_forward, inputs=(weight, bias))
            _jacobian = torch.cat([x.view(-1) for x in _jacobian])
            jacobians.append(_jacobian)

    return jacobians


def compute_influences(
        device: torch.device,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        weight_decay: float,
) -> Tuple[torch.FloatTensor, Dict[str, Union[torch.FloatTensor,
                                              List[torch.FloatTensor]]]]:

    print("Computing Hessian")
    H_train, H_train_pinv = compute_batch_Hessian(
        device=device, model=model,
        data_loader=train_loader,
        weight_decay=weight_decay)

    print("Computing Training Jacobian")
    Js_train = compute_instance_Jacobians(
        device=device, model=model,
        data_loader=train_loader,
        weight_decay=weight_decay)

    print("Computing Test Jacobian")
    Js_test = compute_instance_Jacobians(
        device=device, model=model,
        data_loader=test_loader,
        weight_decay=weight_decay)

    print("Computing Influences")
    with torch.no_grad():
        H_train = H_train.cpu()
        H_train_pinv = H_train_pinv.cpu()
        Js_train = [J.cpu() for J in Js_train]
        Js_test = [J.cpu() for J in Js_test]
        J_train = torch.stack(Js_train, dim=0).cpu()
        J_test = torch.stack(Js_test, dim=0).cpu()
        # Equation:
        # j_test^T H_inv j_train
        # [1, m] x [m, m] x [m, 1] -> [1]
        # J_test^T H_inv J_train
        # --> [n_test, m] x [m, m] x [m, n_train]
        # --> [n_test, n_train]
        # ----------------------------------------
        # Note:
        # `J_test` after stacking is already [n, m]
        # so no more need to transpose `J_test`, instead we need
        # to transpose `J_train` after stacking is also [n, m]
        # so we need to transpose `J_train` instead. Thus,
        # this will look slightly different from the equation above
        influences = - J_test.mm(H_train_pinv).mm(J_train.T)

    return influences, {
        "H_train": H_train,
        "H_train_pinv": H_train_pinv,
        "Js_train": Js_train,
        "Js_test": Js_test,
    }
