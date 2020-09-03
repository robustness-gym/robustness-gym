import torch
import numpy as np
import transformers
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from typing import Union, Dict, Any, List
from transformers import default_data_collator

from influence_utils import parallel
from influence_utils import nn_influence_utils
from experiments.mnli_utils import (
    MNLI2_MODEL_PATH,
    predict,
    create_datasets,
    create_tokenizer_and_model)
from experiments.mnli import (
    WEIGHT_DECAY,
    sort_dict_keys_by_vals)
from experiments.hans_utils import HansHelper
from transformers import TrainingArguments
from experiments.data_utils import (
    glue_output_modes,
    glue_compute_metrics,
    CustomGlueDataset)

NUM_REPLICAS = 3
EXPERIMENT_TYPES = ["most-helpful", "most-harmful", "random"]


def main() -> Dict[str, List[Dict[str, Any]]]:
    task_tokenizer, task_model = create_tokenizer_and_model(
        MNLI2_MODEL_PATH)

    (mnli_train_dataset,
     mnli_eval_dataset) = create_datasets(
        task_name="mnli-2",
        tokenizer=task_tokenizer)

    (hans_train_dataset,
     hans_eval_dataset) = create_datasets(
        task_name="hans",
        tokenizer=task_tokenizer)

    hans_helper = HansHelper(
        hans_train_dataset=hans_train_dataset,
        hans_eval_dataset=hans_eval_dataset)

    output_mode = glue_output_modes["mnli-2"]

    def build_compute_metrics_fn(task_name: str):
        def compute_metrics_fn(p):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Most of these arguments are placeholders
    # and are not really used at all, so ignore
    # the exact values of these.
    trainer = transformers.Trainer(
        model=task_model,
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
        data_collator=default_data_collator,
        train_dataset=mnli_train_dataset,
        eval_dataset=hans_eval_dataset,
        compute_metrics=build_compute_metrics_fn("mnli-2"),
    )

    params_filter = [
        n for n, p in task_model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in task_model.named_parameters()
        if not p.requires_grad]

    output_collections = defaultdict(list)
    with tqdm(total=len(EXPERIMENT_TYPES) * NUM_REPLICAS) as pbar:
        for experiment_type in EXPERIMENT_TYPES:
            for replica_index in range(NUM_REPLICAS):
                outputs_one_experiment = one_experiment(
                    experiment_type=experiment_type,
                    hans_helper=hans_helper,
                    hans_train_dataset=hans_train_dataset,
                    task_model=task_model,
                    params_filter=params_filter,
                    weight_decay_ignores=weight_decay_ignores,
                    trainer=trainer)
                output_collections[experiment_type].append(outputs_one_experiment)

                pbar.update(1)
                pbar.set_description(f"{experiment_type} #{replica_index}")

    return output_collections


def one_experiment(
    experiment_type: str,
    hans_helper: HansHelper,
    hans_train_dataset: CustomGlueDataset,
    task_model: torch.nn.Module,
    params_filter: List[str],
    weight_decay_ignores: List[str],
    trainer: transformers.Trainer
) -> Dict[str, Any]:
    if experiment_type in ["most-harmful", "most-helpful"]:

        hans_eval_heuristic_inputs = hans_helper.sample_batch_of_heuristic(
            mode="eval", heuristic="lexical_overlap", size=128)

        influences, s_test = parallel.compute_influences_parallel(
            # Avoid clash with main process
            device_ids=[1, 2, 3],
            train_dataset=hans_train_dataset,
            batch_size=1,
            model=task_model,
            test_inputs=hans_eval_heuristic_inputs,
            params_filter=params_filter,
            weight_decay=WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=5e-3,
            s_test_scale=1e6,
            s_test_num_samples=1000,
            train_indices_to_include=None,
            return_s_test=True,
            debug=False)

        sorted_indices = sort_dict_keys_by_vals(influences)
        if experiment_type == "most-helpful":
            datapoint_indices = sorted_indices

        if experiment_type == "most-harmful":
            # So that `datapoint_indices[:n]` return the
            # top-n most harmful datapoints
            datapoint_indices = sorted_indices[::-1]

    if experiment_type == "random":
        s_test = None
        influences = None
        hans_eval_heuristic_inputs = None
        # Essentially shuffle the indices
        datapoint_indices = np.random.choice(
            len(hans_train_dataset),
            size=len(hans_train_dataset),
            replace=False)

    loss_collections = {}
    num_datapoints_choices = [1, 10, 100]
    learning_rate_choices = [1e-5, 1e-4, 1e-3]
    for num_datapoints in num_datapoints_choices:
        for learning_rate in learning_rate_choices:
            datapoints = [
                hans_train_dataset[index]
                for index in datapoint_indices[:num_datapoints]]
            batch = default_data_collator(datapoints)
            new_model = pseudo_gradient_step(
                model=task_model,
                inputs=batch,
                learning_rate=learning_rate,
                params_filter=params_filter,
                weight_decay_ignores=weight_decay_ignores)

            for heuristic in ["lexical_overlap", "subsequence", "constituent"]:
                new_model_loss = evaluate_heuristic(
                    hans_helper=hans_helper,
                    heuristic=heuristic,
                    trainer=trainer,
                    model=new_model)

                loss_collections[
                    f"{num_datapoints}-"
                    f"{learning_rate}-"
                    f"{heuristic}"] = new_model_loss
                # print(f"Finished {num_datapoints}-{learning_rate}")

    output_collections = {
        "s_test": s_test,
        "influences": influences,
        "loss": loss_collections,
        "datapoint_indices": datapoint_indices,
        "learning_rates": learning_rate_choices,
        "num_datapoints": num_datapoints_choices,
        "hans_eval_heuristic_inputs": hans_eval_heuristic_inputs,
    }
    return output_collections


def pseudo_gradient_step(
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        learning_rate: float,
        params_filter: List[str],
        weight_decay_ignores: List[str],
) -> torch.nn.Module:

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

    gradients_z = nn_influence_utils.compute_gradients(
        n_gpu=1,
        device=torch.device("cuda"),
        model=model,
        inputs=inputs,
        params_filter=params_filter,
        weight_decay=WEIGHT_DECAY,
        weight_decay_ignores=weight_decay_ignores)

    new_model = deepcopy(model)
    params_to_update = [
        p for name, p in new_model.named_parameters()
        if not any(pfreeze in name for pfreeze in params_to_freeze)]

    with torch.no_grad():
        [p.sub_(learning_rate * grad_z) for p, grad_z in
         zip(params_to_update, gradients_z)]

    return new_model


def evaluate_heuristic(
        hans_helper: HansHelper,
        heuristic: str,
        trainer: transformers.Trainer,
        model: torch.nn.Module,
) -> float:

    _, batch_dataloader = hans_helper.get_dataset_and_dataloader_of_heuristic(
        mode="eval",
        heuristic=heuristic,
        batch_size=1000,
        random=False)

    loss = 0.
    num_examples = 0
    for index, inputs in enumerate(batch_dataloader):
        batch_size = inputs["labels"].shape[0]
        _, _, batch_mean_loss = predict(
            trainer=trainer,
            model=model,
            inputs=inputs)

        num_examples += batch_size
        loss += batch_mean_loss * batch_size

    return loss / num_examples
