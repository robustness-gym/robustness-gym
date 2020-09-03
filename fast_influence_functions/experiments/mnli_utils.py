import torch
from transformers import BertTokenizer
from typing import Tuple, Optional, Union, List


def decode_one_example(tokenizer: BertTokenizer,
                       label_list: List[str],
                       inputs: torch.LongTensor,
                       logits: Optional[torch.FloatTensor] = None
                       ) -> Union[Tuple[str, str], Tuple[str, str, str]]:

    if inputs["input_ids"].shape[0] != 1:
        raise ValueError

    X = tokenizer.decode(inputs["input_ids"][0])
    Y = label_list[inputs["labels"].item()]
    if logits is not None:
        _Y_hat = logits.argmax(dim=-1).item()
        Y_hat = label_list[_Y_hat]
        return X, Y, Y_hat
    else:
        return X, Y


def visualize(tokenizer: BertTokenizer,
              label_list: List[str],
              inputs: torch.LongTensor) -> None:
    X, Y = decode_one_example(
        tokenizer=tokenizer,
        label_list=label_list,
        inputs=inputs,
        logits=None)
    premise, hypothesis = X.split("[CLS]")[1].split("[SEP]")[:2]
    print(f"\tP: {premise.strip()}\n\tH: {hypothesis.strip()}\n\tL: {Y}")
