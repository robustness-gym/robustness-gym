from __future__ import annotations

from typing import Callable, List, Tuple, Union

import numpy as np

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.operation import lookup
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.ops.spacy import SpacyOp
from robustnessgym.slicebuilders.subpopulations.score import ScoreSubpopulation


class NumTokensSubpopulation(ScoreSubpopulation):
    """Subpopulation based on token length."""

    def __init__(
        self,
        intervals: List[Tuple[Union[int, float, str], Union[int, float, str]]],
        reduction_fn: Callable = np.sum,
        **kwargs,
    ):
        super(NumTokensSubpopulation, self).__init__(
            intervals=intervals,
            identifiers=[
                Identifier(
                    _name=self.__class__.__name__,
                    gte=interval[0],
                    lte=interval[1],
                    reduction_fn=reduction_fn,
                )
                for interval in intervals
            ],
            **kwargs,
        )

        # Assign the reduction fn
        self.reduction_fn = reduction_fn

    def score(
        self,
        batch: DataPanel,
        columns: List[str],
        *args,
        **kwargs,
    ) -> np.ndarray:

        # Length of each example, for each column
        try:
            lengths = [
                [len(doc) for doc in lookup(batch, SpacyOp, [col])] for col in columns
            ]
        except AttributeError:
            lengths = [[len(text.split()) for text in batch[col]] for col in columns]

        # Reduction over column key axis
        return self.reduction_fn(np.array(lengths), axis=0)


class NumCharsSubpopulation(ScoreSubpopulation):
    """Subpopulation based on character length."""

    def __init__(
        self,
        intervals: List[Tuple[Union[int, float, str], Union[int, float, str]]],
        reduction_fn: Callable = np.sum,
        **kwargs,
    ):
        super(NumCharsSubpopulation, self).__init__(
            intervals=intervals,
            identifiers=[
                Identifier(
                    _name=self.__class__.__name__,
                    gte=interval[0],
                    lte=interval[1],
                    reduction_fn=reduction_fn,
                )
                for interval in intervals
            ],
            **kwargs,
        )

        # Assign the reduction fn
        self.reduction_fn = reduction_fn

    def score(
        self,
        batch: DataPanel,
        columns: List[str],
        *args,
        **kwargs,
    ) -> np.ndarray:

        # Length of each example, for each column
        lengths = [[len(text) for text in batch[col]] for col in columns]

        # Reduction over column key axis
        return self.reduction_fn(np.array(lengths), axis=0)
