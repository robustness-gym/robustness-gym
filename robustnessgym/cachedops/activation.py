from functools import reduce
from typing import Dict, List

import torch
import torch.nn as nn

from robustnessgym import Dataset
from robustnessgym.core.cachedops import SingleColumnCachedOperation
from robustnessgym.core.decorators import singlecolumn


class ActivationExtractor:
    """Class for extracting activations of a targetted intermediate layer."""

    def __init__(self):
        self.activation = None

    def add_hook(self, module, input, output):
        self.activation = output


class ActivationCachedOp(SingleColumnCachedOperation):
    def __init__(self, model: nn.Module, target_module: str, device: int = None):
        """ A cached operation that runs a forward pass over 
        each example in the dataset and stores model activations in a new column. 
        TODO: test on an NLP model 

        Args:
            model (nn.Module): the torch model from which activations are extracted
            target_module (str): the name of the submodule of `model` (i.e. an
                intermediate layer) that outputs the activations we'd like to extract.
                For nested submodules, specify a path separated by "." (e.g.
                `ActivationCachedOp(model, "block4.conv")`).
            device (int, optional): the device for the forward pass. Defaults to None,
                in which case the CPU is used.
        """
        self.model = model
        self.device = device

        try:
            target_module = _nested_getattr(model, target_module)
        except nn.modules.module.ModuleAttributeError:
            raise ValueError(f"model does not have a submodule {target_module}")

        self.extractor = ActivationExtractor()
        target_module.register_forward_hook(self.extractor.add_hook)

        super(ActivationCachedOp, self).__init__()

    # TODO(sabri): make a proper encoder/decoder pair using torch.save
    @classmethod
    def encode(cls, obj: torch.Tensor) -> torch.Tensor:
        return obj

    @classmethod
    def decode(cls, obj: torch.Tensor) -> torch.Tensor:
        return obj

    def prepare_dataset(
        self,
        dataset: Dataset,
        columns: List[str],
        batch_size: int = 32,
        *args,
        **kwargs,
    ) -> None:

        # First reset the scores
        if self.device is not None:
            self.model.to(self.device)

        # Prepare the dataset
        super(ActivationCachedOp, self).prepare_dataset(
            dataset=dataset,
            columns=columns,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    @singlecolumn
    def apply(self, batch: Dict[str, List], columns: List[str], **kwargs) -> List:
        inputs = batch[columns[0]]

        if self.device is not None:
            inputs = inputs.to(self.device)

        self.model(inputs)
        return self.extractor.activation.cpu().detach()


def _nested_getattr(obj, attr, *args):
    """Get a nested property from an object.

    Example:
    ```
        model = ...
        weights = _nested_getattr(model, "layer4.weights")
    ```
    """
    return reduce(lambda o, a: getattr(o, a, *args), [obj] + attr.split("."))
