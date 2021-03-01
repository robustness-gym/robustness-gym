from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import cytoolz as tz
import numpy as np

from robustnessgym.cachedops.spacy import Spacy
from robustnessgym.core.decorators import prerequisites
from robustnessgym.core.identifier import Identifier
from robustnessgym.slicebuilders.subpopulations.score import ScoreSubpopulation


@prerequisites(Spacy)
class LengthSubpopulation(ScoreSubpopulation):
    """Class to compute subpopulations based on text length."""

    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        reduction_fn: Callable = np.sum,
        *args,
        **kwargs
    ):
        super(LengthSubpopulation, self).__init__(
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
            *args,
            **kwargs,
        )

        # Assign the reduction fn
        self.reduction_fn = reduction_fn

    def score(
        self, batch: Dict[str, List], columns: List[str], *args, **kwargs
    ) -> np.ndarray:
        # Compute the length of each example under each key
        lengths = [
            Spacy.retrieve(
                batch=batch,
                columns=col,
                proc_fns=tz.compose(
                    # Compute lengths (# of words) for each tokenized text in a batch
                    lambda l: np.array([len(t) for t in l]),
                    # Extract tokens using Spacy
                    Spacy.tokens,
                ),
            )
            for col in columns
        ]

        # Reduction over the key axis
        return self.reduction_fn(np.array(lengths), axis=0)
