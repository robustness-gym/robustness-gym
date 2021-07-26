"""Example of running predictions on NLI tasks.

Example:

    # Run squeezebert and bart large (facebook) from HuggingFace on boolq
    $ python prediction.py --dataset boolq \
        --model squeezebert/squeezebert-mnli facebook/bart-large-mnli \
        --output-dir ./
"""
import argparse
import itertools
import os

import pandas as pd
from meerkat import DataPanel
from meerkat.columns.prediction_column import ClassificationOutputColumn

from robustnessgym.core.model import Model
from robustnessgym.tasks.task import BinaryNaturalLanguageInference

# EXAMPLE_DATASETS = [
#     ("boolq", "validation"),
#     ("snli", "validation"),
#     (("glue", "mnli"), "validation_matched"),
# ]

EXAMPLE_DATASETS = [
    ("hans", ["validation"]),
    ("snli", ["validation", "test"]),
    ("anli", ["dev_r1", "test_r1", "dev_r2", "test_r2", "dev_r3", "test_r3"]),
    (
        ("glue", "mnli"),
        [
            "validation_matched",
            "validation_mismatched",
            "test_matched",
            "test_mismatched",
        ],
    ),
]
EXAMPLE_DATASETS_TO_SPLITS = {x[0]: x[1] for x in EXAMPLE_DATASETS}

EXAMPLE_MODELS = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",
    "textattack/bert-base-uncased-snli",
    "textattack/distilbert-base-cased-snli",
    "textattack/bert-base-uncased-MNLI",
    "textattack/albert-base-v2-snli",
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
    "microsoft/deberta-large-mnli",
    "microsoft/deberta-v2-xxlarge-mnli",
    "typeform/mobilebert-uncased-mnli",
    "huggingface/distilbert-base-uncased-finetuned-mnli",
    "prajjwal1/albert-base-v1-mnli",
    "prajjwal1/bert-tiny-mnli",
    "cross-encoder/nli-deberta-base",
    "squeezebert/squeezebert-mnli",
    "squeezebert/squeezebert-mnli-headless",
]


def run_nli(
    dataset_info,
    model_name,
    debug: bool = False,
    output_dir: str = None,
):
    """Compute and save logits for natural language inference tasks.

    Args:
        dataset_info (Tuple[Union[str, Tuple[str]]]): Dataset name and split.
        model_name (str): The HuggingFace model to use.
        debug (bool, optional): If ``True``, runs in debug mode.
            This runs inference on the first 10 examples in the dataset.

    Returns:
        ClassificationOutputColumn: The output column with DataPanel.
    """

    def fn(dp):
        """Computes logits over the datapanel."""
        if "passage" in dp:
            data = {"col1": dp["passage"].data, "col2": dp["question"].data}
        else:
            data = {"col1": dp["premise"].data, "col2": dp["hypothesis"].data}

        output = model.predict_batch(data, ["col1", "col2"])  # noqa: F821
        return output["logits"].cpu()

    # Load the dataset as a DataPanel.
    dataset_name, split = dataset_info
    if isinstance(dataset_name, str):
        dataset_name = (dataset_name,)
    dp = DataPanel.load_huggingface(*dataset_name, split=split)
    if debug:
        dp = dp[:10]

    # Load the model.
    model = Model.huggingface(
        model_name, task=BinaryNaturalLanguageInference(), is_classifier=True
    )

    # Compute predictions on the data.
    out = dp.map(
        fn,
        batch_size=5,
        is_batched_fn=True,
        output_type=ClassificationOutputColumn,
        pbar=True,
    )
    out = ClassificationOutputColumn(logits=out)

    del model

    model_name = model_name.replace("/", "-")
    if output_dir:
        dirname = (
            f"{output_dir}/nli-preds/" f"{'_'.join(dataset_name)}-{split}-{model_name}"
        )
        out.write(dirname)

    # TO LOAD:
    # ClassificationOutputColumn.read(dirname)

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        "Run NLI tasks in RobustnessGym with built-in HuggingFace models."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        type=str,
        choices=[
            "_".join(x[0]) if isinstance(x[0], tuple) else x[0]
            for x in EXAMPLE_DATASETS
        ],
        help="Dataset(s) to use",
        required=True,
    )
    parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        choices=EXAMPLE_MODELS,
        help="Model(s) to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to save predictions for easy loading",
        default="./",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [tuple(x.split("_")) for x in args.dataset]
    datasets = [x if len(x) > 1 else x[0] for x in datasets]
    all_models = args.model
    output_dir = args.output_dir
    debug = args.debug

    if all_models is None:
        all_models = EXAMPLE_MODELS

    configs = list(itertools.product(datasets, all_models))
    outs = []
    for idx, (ds, model) in enumerate(configs):
        for split in EXAMPLE_DATASETS_TO_SPLITS[ds]:
            print(
                f"***** ({idx}/{len(configs)}: "
                f"Dataset: {ds} - Split: {split} - Model: {model}) *****"
            )
            ds = (ds, split)
            try:
                run_nli(ds, model, debug=debug, output_dir=output_dir)
                outs.append(
                    {
                        "dataset": str(ds),
                        "model": model,
                        "status": "Success",
                        "Reason": None,
                    }
                )
            except Exception as e:
                print(e)
                outs.append(
                    {
                        "dataset": str(ds),
                        "model": model,
                        "status": "Failed",
                        "Reason": str(e),
                    }
                )

    if output_dir:
        run_status = pd.DataFrame(outs)
        run_status.to_csv(os.path.join(output_dir, "run_status.csv"))


if __name__ == "__main__":
    main()
