import itertools

import pandas as pd
from meerkat import DataPanel
from meerkat.columns.prediction_column import ClassificationOutputColumn

from robustnessgym.core.model import Model
from robustnessgym.tasks.task import ExtractiveQuestionAnswering


def run(dataset_info, model_name, debug=False):
    def fn(dp):
        # TODO(karan): Uncomment & add correct code here.
        # batch = {"col1": dp["context"].data, "col2": dp["question"].data}
        # input_cols = ["context", "question"]
        #
        # input_batch = model.encode_batch(batch, input_cols)
        # import cytoolz as tz
        # import torch
        # input_batch = tz.valmap(
        #     lambda v: torch.tensor(v).to(device=model.device),
        #     input_batch
        # )
        #
        # with torch.no_grad():
        #     outputs = model.model(**input_batch)
        #
        # outputs = (torch.argmax(outputs.start_logits).item(),
        #            torch.argmax(outputs.end_logits).item())
        #
        # token_id = input_batch['input_ids'][0][outputs[0]: outputs[1]]
        # answer = model.tokenizer.decode(token_id)
        # return answer
        pass

    dataset_name, split = dataset_info
    if isinstance(dataset_name, str):
        dataset_name = (dataset_name,)
    dp = DataPanel.load_huggingface(*dataset_name, split=split)
    if debug:
        dp = dp[:10]

    model = Model.huggingface(
        model_name,
        task=ExtractiveQuestionAnswering(),
        is_classifier=False,
    )

    out = dp.map(
        fn,
        batch_size=5,
        is_batched_fn=True,
        # output_type=ClassificationOutputColumn,
        pbar=True,
    )

    out = ClassificationOutputColumn(logits=out)

    del model

    dirname = f"./{'_'.join(dataset_name)}-{model_name}"
    out.write(dirname)

    # TO LOAD:
    # ClassificationOutputColumn.read(dirname)

    return out


DATASETS = [
    ("squad", "validation"),
]

# 'squad' 'validation'

MODELS = [
    "distilbert-base-cased-distilled-squad",
]

ds = DATASETS[0]
model = MODELS[0]

outs = []
for ds, model in itertools.product(DATASETS, MODELS):
    try:
        run(ds, model, debug=True)
        outs.append(
            {"dataset": str(ds), "model": model, "status": "Success", "Reason": None}
        )
    except Exception as e:
        print(e)
        outs.append(
            {"dataset": str(ds), "model": model, "status": "Failed", "Reason": str(e)}
        )

success = pd.DataFrame(outs)
