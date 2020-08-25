import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import transformers
from tqdm import tqdm
from copy import deepcopy
from contexttimer import Timer
from collections import defaultdict
from influence_utils import faiss_utils
from influence_utils import nn_influence_utils
from typing import List, Dict, Tuple, Optional, Union, Any

from experiments import constants
from experiments import misc_utils


def run_full_influence_functions(
        num_examples_to_test: int,
        s_test_num_samples: int = 1000
) -> Dict[int, Dict[str, Any]]:
    tokenizer, model = misc_utils.create_tokenizer_and_model(
        constants.MNLI_MODEL_PATH)

    (mnli_train_dataset,
     mnli_eval_dataset) = misc_utils.create_datasets(
        task_name="mnli",
        tokenizer=tokenizer)

    batch_train_data_loader = misc_utils.get_dataloader(
        mnli_train_dataset,
        batch_size=128,
        random=True)

    instance_train_data_loader = misc_utils.get_dataloader(
        mnli_train_dataset,
        batch_size=1,
        random=False)

    eval_instance_data_loader = misc_utils.get_dataloader(
        dataset=mnli_eval_dataset,
        batch_size=1,
        random=False)

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    model.cuda()
    outputs_collections = {}
    for test_index, test_inputs in enumerate(eval_instance_data_loader):
        print(f"Running #{test_index}")
        if test_index >= num_examples_to_test:
            break

        with Timer() as timer:
            influences, _, s_test = nn_influence_utils.compute_influences(
                n_gpu=1,
                device=torch.device("cuda"),
                batch_train_data_loader=batch_train_data_loader,
                instance_train_data_loader=instance_train_data_loader,
                model=model,
                test_inputs=test_inputs,
                params_filter=params_filter,
                weight_decay=constants.WEIGHT_DECAY,
                weight_decay_ignores=weight_decay_ignores,
                s_test_damp=5e-3,
                s_test_scale=1e4,
                s_test_num_samples=s_test_num_samples,
                train_indices_to_include=None,
                s_test_iterations=1,
                precomputed_s_test=None)

            outputs_collections[test_index] = {
                "influences": influences,
                "s_test": s_test,
                "time": timer.elapsed}

    return outputs_collections


def get_influences(
        k: int,
        model: torch.nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        batch_train_data_loader: torch.utils.data.DataLoader,
        instance_train_data_loader: torch.utils.data.DataLoader,
        device_ids: Optional[List[int]] = None,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
) -> Tuple[Dict[int, float], List[torch.FloatTensor]]:

    faiss_index = faiss_utils.FAISSIndex(768, "Flat")
    faiss_index.load(constants.MNLI_FAISS_INDEX_PATH)
    print(f"Loaded FAISS index with {len(faiss_index)} entries")

    test_features = misc_utils.compute_BERT_CLS_feature(model, **test_inputs)
    test_features = test_features.cpu().detach().numpy()
    KNN_distances, KNN_indices = faiss_index.search(
        k=k, queries=test_features)

    # Make sure indices are sorted according to distances
    # KNN_distances[(
    #     KNN_indices.squeeze(axis=0)[
    #         np.argsort(KNN_distances.squeeze(axis=0))
    #     ] != KNN_indices)]

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    if device_ids is None:
        (influences,
         train_inputs_collections,
         s_test) = nn_influence_utils.compute_influences(
            n_gpu=1,
            device=torch.device("cuda"),
            model=model,
            test_inputs=test_inputs,
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores,
            s_test_scale=1000,
            s_test_num_samples=300,
            precomputed_s_test=precomputed_s_test,
            train_indices_to_include=KNN_indices)
    else:
        raise ValueError("Deprecated")

    return influences, s_test


def main(
    label_list: List[str],
    task_model: torch.nn.Module,
    imitator_model: torch.nn.Module,
    trainer: transformers.Trainer,
    test_data_point_indices: List[int],
    batch_train_data_loader: torch.utils.data.DataLoader,
    instance_train_data_loader: torch.utils.data.DataLoader,
    instance_eval_data_loader: torch.utils.data.DataLoader,
    sample_size: int = 10,
    num_nearest_neighbors: int = 10000,
    finetune_using_ground_truth_label: bool = False
) -> List[Dict[str, Any]]:

    train_inputs_collections = torch.load(
        constants.MNLI_TRAIN_INPUT_COLLECTIONS_PATH)

    neutral_examples = []
    entailment_examples = []
    contradiction_examples = []
    for i in range(len(train_inputs_collections)):
        label = label_list[train_inputs_collections[i]["labels"]]
        if label == "neutral":
            neutral_examples.append(i)
        if label == "entailment":
            entailment_examples.append(i)
        if label == "contradiction":
            contradiction_examples.append(i)

    outputs_collections = []
    for i, test_inputs in enumerate(instance_eval_data_loader):
        if i not in test_data_point_indices:
            continue

        start_time = time.time()
        imitator_test_inputs = experimental_make_imitator_inputs(
            trainer=trainer, task_model=task_model, inputs=test_inputs)
        # if labels[0] != logits.argmax(axis=1)[0]:
        #     break
        influences, _ = get_influences(
            k=num_nearest_neighbors,
            model=task_model,
            test_inputs=test_inputs,
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader)

        data_indices = (
            np.random.choice(neutral_examples,
                             size=sample_size,
                             replace=False).tolist() +  # noqa
            np.random.choice(entailment_examples,
                             size=sample_size,
                             replace=False).tolist() +  # noqa
            np.random.choice(contradiction_examples,
                             size=sample_size,
                             replace=False).tolist() +  # noqa
            misc_utils.sort_dict_keys_by_vals(influences)[:sample_size] +  # noqa
            misc_utils.sort_dict_keys_by_vals(influences)[-sample_size:]
        )

        data_tags = (
            ["random-neutral" for _ in range(sample_size)] +  # noqa
            ["random-entailment" for _ in range(sample_size)] +  # noqa
            ["random-contradiction" for _ in range(sample_size)] +  # noqa
            ["most-negative-influential" for _ in range(sample_size)] +  # noqa
            ["most-positive-influential" for _ in range(sample_size)]
        )

        learning_rates = np.logspace(-5, -2.5, 50)
        losses = compute_new_imitator_losses(
            trainer=trainer,
            tags=data_tags,
            indices=data_indices,
            task_model=task_model,
            imitator_model=imitator_model,
            learning_rates=learning_rates,
            imitator_test_inputs=imitator_test_inputs,
            train_inputs_collections=train_inputs_collections,
            finetune_using_ground_truth_label=finetune_using_ground_truth_label)

        outputs_collections.append({
            "index": i,
            "losses": losses,
            "influences": influences,
            "test_inputs": test_inputs,
            "learning_rates": learning_rates,
            "imitator_test_inputs": imitator_test_inputs
        })

        end_time = time.time()
        print(f"#{len(outputs_collections)}/{len(outputs_collections)}: "
              f"Elapsed {(end_time - start_time) / 60:.2f}")

    return outputs_collections


def compute_new_imitator_losses(
        indices: List[int],
        tags: List[str],
        task_model: torch.nn.Module,
        imitator_model: torch.nn.Module,
        trainer: transformers.Trainer,
        learning_rates: Union[np.ndarray, List[float]],
        imitator_test_inputs: Dict[str, torch.Tensor],
        train_inputs_collections: List[Dict[str, torch.Tensor]],
        finetune_using_ground_truth_label: bool = False,
) -> Dict[str, List[List[float]]]:

    params_filter = [
        n for n, p in imitator_model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in imitator_model.named_parameters()
        if not p.requires_grad]

    params_to_freeze = [
        "bert.embeddings.",
        "bert.encoder.layer.0.",
        "bert.encoder.layer.1.",
        "bert.encoder.layer.2.",
        "bert.encoder.layer.3.",
        "bert.encoder.layer.4.",
        "bert.encoder.layer.5.",
        "bert.encoder.layer.6.",
        "bert.encoder.layer.7.",
        "bert.encoder.layer.8.",
        "bert.encoder.layer.9.",
    ]

    losses = defaultdict(list)
    for index, tag in zip(tqdm(indices), tags):
        if finetune_using_ground_truth_label is True:
            imitator_train_inputs = train_inputs_collections[index]
        else:
            imitator_train_inputs = experimental_make_imitator_inputs(
                trainer=trainer,
                task_model=task_model,
                inputs=train_inputs_collections[index])

        helpful_grad_z = nn_influence_utils.compute_gradients(
            n_gpu=1,
            device=torch.device("cuda"),
            model=imitator_model,
            inputs=imitator_train_inputs,
            params_filter=params_filter,
            weight_decay=WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores)

        _losses = []
        for lr in learning_rates:
            new_imitator_model = deepcopy(imitator_model)
            params_to_update = [
                p for name, p in new_imitator_model.named_parameters()
                if not any(pfreeze in name for pfreeze in params_to_freeze)]

            with torch.no_grad():
                [p.sub_(lr * grad_z) for p, grad_z in
                 zip(params_to_update, helpful_grad_z)]

            _, _, imitator_loss = misc_utils.predict(
                trainer=trainer,
                model=new_imitator_model,
                inputs=imitator_test_inputs)
            _losses.append(imitator_loss)

        losses[tag].append(_losses)

    return losses


def experimental_make_imitator_inputs(
        trainer: transformers.Trainer,
        task_model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    logits, _, _ = misc_utils.predict(
        trainer=trainer, model=task_model, inputs=inputs)
    imitator_inputs = deepcopy(inputs)
    imitator_inputs["labels"] = torch.tensor(logits.argmax(axis=1))
    return imitator_inputs


def plot_Xs_and_Ys_dict(
        Xs: List[float],
        Ys_dict: Dict[str, List[List[float]]]
) -> None:
    # plt.rcParams["figure.figsize"] = (10, 10)
    color_map = {
        "random-neutral": "grey",
        "random-entailment": "salmon",
        "random-contradiction": "skyblue",
        "most-positive-influential": "darkred",
        "most-negative-influential": "steelblue"}

    legends = []
    for tag in Ys_dict.keys():
        if tag not in color_map.keys():
            raise ValueError

        legends.append(tag)
        color = color_map[tag]
        data = np.array(Ys_dict[tag])
        is_random_data_point = "random" in tag

        if data.shape[0] != 1:
            data_mean = data.mean(axis=0)
            data_max = data.max(axis=0)
            data_min = data.min(axis=0)
            # data_std = data.std(axis=0)
            plt.plot(Xs, data_mean,
                     color=color,
                     linestyle=("--" if is_random_data_point else None))

            # plt.fill_between(Xs,
            #                  data_mean + 1. * data_std,
            #                  data_mean - 1. * data_std,
            #                  color=color,
            #                  alpha=0.1 if is_random_data_point else 0.2)
            plt.fill_between(Xs, data_max, data_min,
                             alpha=0.1,
                             color=color)
        else:
            plt.plot(Xs, data[0, ...], color=color)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.legend(legends, fontsize=15)
    plt.title("Loss of the Imitator Model", fontsize=30)
    # plt.savefig("./20200719-fig1.pdf")
