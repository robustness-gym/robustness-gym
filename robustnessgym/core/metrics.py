import re
import statistics
from typing import Sequence, Union

import nltk
import pytorch_lightning.metrics.functional as lightning_metrics
import torch
from rouge_score import rouge_scorer


# TODO Refactor into separate class for each metric
# TODO change signature of compute_metric
def compute_metric(
    metric: str,
    predictions: Union[Sequence, torch.Tensor],
    labels: Union[Sequence, torch.Tensor],
    num_classes: int,
):
    """Compute metric given predictions and target labels
    Args:
        metric: name of metric
        predictions: A sequence of predictions (rouge metrics) or a torch Tensor
        (other metrics) containing predictions
        labels: A sequence of labels (rouge metrics) or a torch Tensor (other metrics)
        containing target labels
        num_classes: number of classes
    """

    if metric == "accuracy":
        # Calculate the accuracy
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.Tensor(predictions)
        if not isinstance(labels, torch.Tensor):
            labels = torch.Tensor(labels)
        score = lightning_metrics.accuracy(
            pred=predictions,
            target=labels,
            num_classes=num_classes,
        ).item()
    elif metric == "f1":
        # Calculate the f1
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.Tensor(predictions)
        if not isinstance(labels, torch.Tensor):
            labels = torch.Tensor(labels)
        score = lightning_metrics.f1_score(
            pred=predictions,
            target=labels,
            num_classes=num_classes,
        ).item()
    elif metric in ("Rouge-1", "Rouge-2", "Rouge-L"):
        # Calculate rouge scores
        if metric == "Rouge-1":
            metric_id = "rouge1"
        elif metric == "Rouge-2":
            metric_id = "rouge2"
        else:
            metric_id = "rougeLsum"
        scorer = rouge_scorer.RougeScorer([metric_id], use_stemmer=True)
        # TODO Remove summarizaton-specific 'format_summary' call
        # TODO Don't call scorer.score separately for each metric
        score = statistics.mean(
            scorer.score(format_summary(reference), format_summary(pred))[
                metric
            ].fmeasure
            for reference, pred in zip(labels, predictions)
        )

    elif metric == "class_dist":
        # Calculate class distribution
        if not isinstance(labels, torch.Tensor):
            labels = torch.Tensor(labels)
        score = (
            lightning_metrics.to_onehot(tensor=labels, num_classes=num_classes)
            .double()
            .mean(dim=0)
            .tolist()
        )

    elif metric == "pred_dist":
        # Calculate predicted class distribution
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.Tensor(predictions)
        score = (
            lightning_metrics.to_onehot(tensor=predictions, num_classes=num_classes)
            .double()
            .mean(dim=0)
            .tolist()
        )
    else:
        raise NotImplementedError

    return score


def format_summary(x: str) -> str:
    """Format summary text for computing rouge."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    return "\n".join(nltk.sent_tokenize(x))
