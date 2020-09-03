import torch
import numpy as np
# from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from typing import Tuple, Optional, Union, Any, Dict, List
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    GlueDataTrainingArguments,
    Trainer,
    DataCollator,
    default_data_collator)

from influence_utils import glue_utils
from experiments import constants
from experiments.data_utils import CustomGlueDataset


def sort_dict_keys_by_vals(d: Dict[int, float]) -> List[int]:
    sorted_items = sorted(list(d.items()),
                          key=lambda pair: pair[1])
    return [pair[0] for pair in sorted_items]


def compute_BERT_CLS_feature(
    model,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
) -> torch.FloatTensor:
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
        Labels for computing the sequence classification/regression loss.
        Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
        If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    if model.training is True:
        raise ValueError

    outputs = model.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    pooled_output = outputs[1]

    return model.dropout(pooled_output)


def create_tokenizer_and_model(
        model_name_or_path: str,
        freeze_parameters: bool = True
) -> Tuple[BertTokenizer, BertForSequenceClassification]:
    if model_name_or_path is None:
        raise ValueError
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

    model.eval()
    if freeze_parameters is True:
        glue_utils.freeze_BERT_parameters(model)

    return tokenizer, model


def create_datasets(
        task_name: str,
        tokenizer: BertTokenizer,
        data_dir: Optional[str] = None
) -> Tuple[CustomGlueDataset, CustomGlueDataset]:
    if task_name not in ["mnli", "mnli-2", "hans"]:
        raise ValueError(f"Unrecognized task {task_name}")

    if data_dir is None:
        if task_name in ["mnli", "mnli-2"]:
            data_dir = constants.GLUE_DATA_DIR
        if task_name in ["hans"]:
            data_dir = constants.HANS_DATA_DIR

    data_args = GlueDataTrainingArguments(
        task_name=task_name,
        data_dir=data_dir,
        max_seq_length=128)

    train_dataset = CustomGlueDataset(
        args=data_args,
        tokenizer=tokenizer,
        mode="train")

    eval_dataset = CustomGlueDataset(
        args=data_args,
        tokenizer=tokenizer,
        mode="dev")

    return train_dataset, eval_dataset


def predict(trainer: Trainer,
            model: torch.nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:

    if trainer.args.past_index >= 0:
        raise ValueError

    has_labels = any(
        inputs.get(k) is not None for k in
        ["labels", "lm_labels", "masked_lm_labels"])

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(trainer.args.device)

    step_eval_loss = None
    with torch.no_grad():
        outputs = model(**inputs)
        if has_labels:
            step_eval_loss, logits = outputs[:2]
        else:
            logits = outputs[0]

    preds = logits.detach()
    preds = preds.cpu().numpy()
    if inputs.get("labels") is not None:
        label_ids = inputs["labels"].detach()
        label_ids = label_ids.cpu().numpy()

    if step_eval_loss is not None:
        step_eval_loss = step_eval_loss.mean().item()

    return preds, label_ids, step_eval_loss


def get_dataloader(dataset: CustomGlueDataset,
                   batch_size: int,
                   random: bool = False,
                   data_collator: Optional[DataCollator] = None
                   ) -> DataLoader:
    if data_collator is None:
        data_collator = default_data_collator

    if random is True:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return data_loader
