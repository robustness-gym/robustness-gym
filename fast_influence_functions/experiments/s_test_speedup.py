import sys
import torch
from contexttimer import Timer
from typing import List, Dict, Any
from transformers import GlueDataset

from influence_utils import parallel
from influence_utils import faiss_utils
from influence_utils import nn_influence_utils
from influence_utils.nn_influence_utils import compute_s_test
from experiments import constants
from experiments import misc_utils


def one_experiment(
        model: torch.nn.Module,
        train_dataset: GlueDataset,
        test_inputs: Dict[str, torch.Tensor],
        batch_size: int,
        random: bool,
        n_gpu: int,
        device: torch.device,
        damp: float,
        scale: float,
        num_samples: int,
) -> List[torch.Tensor]:

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    # Make sure each dataloader is re-initialized
    batch_train_data_loader = misc_utils.get_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        random=random)

    s_test = compute_s_test(
        n_gpu=n_gpu,
        device=device,
        model=model,
        test_inputs=test_inputs,
        train_data_loaders=[batch_train_data_loader],
        params_filter=params_filter,
        weight_decay=constants.WEIGHT_DECAY,
        weight_decay_ignores=weight_decay_ignores,
        damp=damp,
        scale=scale,
        num_samples=num_samples)

    return [X.cpu() for X in s_test]


def main(
    num_examples_to_test: int = 5,
    num_repetitions: int = 1,
) -> List[Dict[str, Any]]:

    task_tokenizer, task_model = misc_utils.create_tokenizer_and_model(
        constants.MNLI_MODEL_PATH)
    train_dataset, eval_dataset = misc_utils.create_datasets(
        task_name="mnli",
        tokenizer=task_tokenizer)
    eval_instance_data_loader = misc_utils.get_dataloader(
        dataset=eval_dataset,
        batch_size=1,
        random=False)

    task_model.cuda()
    output_collections = []
    for test_index, test_inputs in enumerate(eval_instance_data_loader):
        if test_index >= num_examples_to_test:
            break

        for k, v in test_inputs.items():
            if isinstance(v, torch.Tensor):
                test_inputs[k] = v.to(torch.device("cuda"))

        # with batch-size 128, 1500 iterations is enough
        for num_samples in range(700, 1300 + 1, 100):  # 7 choices
            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:  # 8 choices
                for repetition in range(num_repetitions):
                    print(f"Running #{test_index} "
                          f"N={num_samples} "
                          f"B={batch_size} "
                          f"R={repetition} takes ...", end=" ")
                    with Timer() as timer:
                        s_test = one_experiment(
                            model=task_model,
                            train_dataset=train_dataset,
                            test_inputs=test_inputs,
                            batch_size=batch_size,
                            random=True,
                            n_gpu=1,
                            device=torch.device("cuda"),
                            damp=5e-3,
                            scale=1e4,
                            num_samples=num_samples)
                        time_elapsed = timer.elapsed
                        print(f"{time_elapsed:.2f} seconds")

                    output_collections.append({
                        "test_index": test_index,
                        "num_samples": num_samples,
                        "batch_size": batch_size,
                        "repetition": repetition,
                        "s_test": s_test,
                        "time_elapsed": time_elapsed,
                    })

    return output_collections


def speed_test(version: str, num_examples_to_test: int) -> Dict:
    print(f"Running Speed Test of {version} with {num_examples_to_test} examples")
    if "v3" in version:
        # We want to make sure no GPU resources are used
        # in the main process, since child processes will
        # use GPUs as well, plus, this computation is not
        # very time-consuming.
        device = torch.device("cpu")
    else:
        # Otherwise, use CUDA for the main process
        device = torch.device("cuda")

    tokenizer, model = misc_utils.create_tokenizer_and_model(
        "/export/home/Experiments/20200706/")
    train_dataset, eval_dataset = misc_utils.create_datasets(
        tokenizer=tokenizer)
    eval_instance_data_loader = misc_utils.get_dataloader(
        dataset=eval_dataset,
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

    model.to(device)
    faiss_index = faiss_utils.FAISSIndex(768, "Flat")
    faiss_index.load(constants.MNLI_FAISS_INDEX_PATH)
    outputs_collections = {}
    for test_index, test_inputs in enumerate(eval_instance_data_loader):
        if test_index >= num_examples_to_test:
            break

        for k, v in test_inputs.items():
            if isinstance(v, torch.Tensor):
                test_inputs[k] = v.to(device)

        test_features = misc_utils.compute_BERT_CLS_feature(model, **test_inputs)
        test_features = test_features.cpu().detach().numpy()
        for k in [1000, 10000]:
            KNN_distances, KNN_indices = faiss_index.search(
                k=k, queries=test_features)

            # V3: Using KNN, fast `s_test`, parallel
            if version == "v3":
                with Timer() as timer:
                    influences, s_test = parallel.compute_influences_parallel(
                        device_ids=[0, 1, 2, 3],
                        train_dataset=train_dataset,
                        batch_size=1,
                        model=model,
                        test_inputs=test_inputs,
                        params_filter=params_filter,
                        weight_decay=constants.WEIGHT_DECAY,
                        weight_decay_ignores=weight_decay_ignores,
                        s_test_damp=5e-3,
                        s_test_scale=1e4,
                        s_test_num_samples=1000,
                        train_indices_to_include=KNN_indices,
                        return_s_test=False,
                        debug=False)

                    time_elapsed = timer.elapsed

            # V3 but also return `s_test`
            if version == "v3-stest":
                with Timer() as timer:
                    influences, s_test = parallel.compute_influences_parallel(
                        device_ids=[0, 1, 2, 3],
                        train_dataset=train_dataset,
                        batch_size=1,
                        model=model,
                        test_inputs=test_inputs,
                        params_filter=params_filter,
                        weight_decay=constants.WEIGHT_DECAY,
                        weight_decay_ignores=weight_decay_ignores,
                        s_test_damp=5e-3,
                        s_test_scale=1e4,
                        s_test_num_samples=1000,
                        train_indices_to_include=KNN_indices,
                        return_s_test=True,
                        debug=False)

                    time_elapsed = timer.elapsed

            # V1: Using KNN, fast `s_test`
            if version == "v2":
                model.cuda()
                batch_train_data_loader = misc_utils.get_dataloader(
                    train_dataset,
                    batch_size=1,
                    random=True)

                instance_train_data_loader = misc_utils.get_dataloader(
                    train_dataset,
                    batch_size=1,
                    random=False)

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
                        s_test_num_samples=1000,
                        train_indices_to_include=KNN_indices,
                        s_test_iterations=4,
                        precomputed_s_test=None)

                    time_elapsed = timer.elapsed

            # V1: Using KNN
            if version == "v1":
                model.cuda()
                batch_train_data_loader = misc_utils.get_dataloader(
                    train_dataset,
                    batch_size=128,
                    random=True)

                instance_train_data_loader = misc_utils.get_dataloader(
                    train_dataset,
                    batch_size=1,
                    random=False)

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
                        s_test_num_samples=1000,
                        train_indices_to_include=KNN_indices,
                        s_test_iterations=1,
                        precomputed_s_test=None)

                    time_elapsed = timer.elapsed

            outputs_collections[f"{version}-{test_index}-{k}"] = {
                "influences": influences,
                "s_test": s_test,
                "time": time_elapsed,
            }

    return outputs_collections


if __name__ == "__main__":
    outputs_collections = speed_test(sys.argv[1], int(sys.argv[2]))
    torch.save(outputs_collections,
               f"/export/home/Experiments/20200728/"
               f"{sys.argv[1]}-{int(sys.argv[2])}")
