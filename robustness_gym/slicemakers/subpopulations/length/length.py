from __future__ import annotations

from typing import List, Dict, Tuple, Callable

import numpy as np
import cytoolz as tz

from robustness_gym.cached_ops.spacy.spacy import Spacy

from robustness_gym.identifier import Identifier
from robustness_gym.slicemakers.subpopulations.score.score import HasScore


class HasLength(HasScore,
                Spacy):

    def __init__(self,
                 intervals: List[Tuple[int, int]],
                 reduction_fn: Callable = np.sum,
                 *args,
                 **kwargs
                 ):
        super(HasLength, self).__init__(
            identifiers=[Identifier(name=self.__class__.__name__,
                                    gte=interval[0],
                                    lte=interval[1],
                                    reduction_fn=reduction_fn)
                         for interval in intervals],
            *args,
            **kwargs,
        )

        # Assign the reduction fn
        self.reduction_fn = reduction_fn

    def score(self,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> np.ndarray:
        # Compute the length of each example under each key
        lengths = [
            Spacy.retrieve(batch=batch,
                           keys=[key],
                           proc_fns=tz.compose(lambda l: [len(t) for t in l], Spacy.tokens))[key]
            for key in keys
        ]

        lengths = [
            np.array([len(tokens) for tokens in
                      np.array([len(cache['Spacy'][key]['tokens']) for cache in batch['cache']])
                      for key in keys
                      ]

        # Reduction over the key axis

    return self.reduction_fn(np.array(lengths), axis=0)
