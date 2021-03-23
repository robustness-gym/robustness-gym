import re
import statistics
from typing import Callable, Sequence, Union

try:
    import nltk
except ImportError:
    _nltk_available = False
else:
    _nltk_available = True
import numpy as np
import pytorch_lightning.metrics.functional as lightning_metrics
import torch

try:
    from rouge_score import rouge_scorer
except ImportError:
    _rouge_score_available = False
    rouge_scorer = None
else:
    _rouge_score_available = True
from sklearn.metrics import accuracy_score, f1_score


def get_metric(name: str) -> Callable:
    """Get metrics from string names."""
    if name == "accuracy":
        return accuracy
    elif name == "f1":
        return f1
    elif name == "f1_micro":
        return f1_micro
    elif name == "f1_macro":
        return f1_macro
    else:
        raise NotImplementedError(f"Metric name {name} not recognized.")


def accuracy(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
):
    """Calculate accuracy."""
    return accuracy_score(y_true=labels, y_pred=predictions)


def f1(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
):
    """Calculate F1 score for binary classification."""
    return f1_score(y_true=labels, y_pred=predictions)


def f1_micro(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
):
    """Calculate micro F1 score for multi-class classification."""
    return f1_score(y_true=labels, y_pred=predictions, average="micro")


def f1_macro(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
):
    """Calculate macro F1 score for multi-class classification."""
    return f1_score(y_true=labels, y_pred=predictions, average="macro")


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

    # Classification metrics
    if metric == "accuracy":
        return accuracy(predictions=predictions, labels=labels)
    elif metric == "f1":
        return f1(predictions=predictions, labels=labels)
    elif metric == "f1":
        return f1(predictions=predictions, labels=labels)
    elif metric == "f1_micro":
        return f1_micro(predictions=predictions, labels=labels)
    elif metric == "f1_macro":
        return f1_macro(predictions=predictions, labels=labels)

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
