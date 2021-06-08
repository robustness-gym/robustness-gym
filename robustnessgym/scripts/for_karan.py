import itertools

import pandas as pd
from mosaic import DataPanel
from mosaic.columns.prediction_column import ClassificationOutputColumn

from robustnessgym.core.model import Model
from robustnessgym.tasks.task import BinaryNaturalLanguageInference


def run(dataset_info, model_name, debug=False):
    def fn(dp):
        if "passage" in dp:
            data = {"col1": dp["passage"].data, "col2": dp["question"].data}
        else:
            data = {"col1": dp["premise"].data, "col2": dp["hypothesis"].data}

        output = model.predict_batch(data, ["col1", "col2"])
        return output["logits"].cpu()

    dataset_name, split = dataset_info
    if isinstance(dataset_name, str):
        dataset_name = (dataset_name,)
    dp = DataPanel.load_huggingface(*dataset_name, split=split)
    if debug:
        dp = dp[:10]

    model = Model.huggingface(
        model_name,
        task=BinaryNaturalLanguageInference(),
        is_classifier=True,
    )

    out = dp.map(
        fn,
        batch_size=5,
        is_batched_fn=True,
        output_type=ClassificationOutputColumn,
        pbar=True
    )

    out = ClassificationOutputColumn(logits=out)

    del model

    dirname = f"./{'_'.join(dataset_name)}-{model_name}"
    out.write(dirname)

    # TO LOAD:
    # ClassificationOutputColumn.read(dirname)

    return out


DATASETS = [
    ("boolq", "validation"),
    ("snli", "validation"),
    (('glue', 'mnli'), "validation_matched"),
]

MODELS = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "textattack/bert-base-uncased-snli",
    "facebook/bart-large-mnli",
    "textattack/bert-base-uncased-MNLI",
    "huggingface/distilbert-base-uncased-finetuned-mnli",
    "prajjwal1/albert-base-v1-mnli",
    "cross-encoder/nli-deberta-base",
    "squeezebert/squeezebert-mnli",
]

ds = DATASETS[0]
model = MODELS[0]

outs = []
for ds, model in itertools.product(DATASETS, MODELS):
    try:
        run(ds, model, debug=True)
        outs.append(
            {
                "dataset": str(ds),
                "model": model,
                "status": "Success",
                "Reason": None
            }
        )
    except Exception as e:
        print(e)
        outs.append(
            {
                "dataset": str(ds),
                "model": model,
                "status": "Failed",
                "Reason": str(e)
            }
        )

success = pd.DataFrame(outs)
