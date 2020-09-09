from typing import List, Tuple, Dict, Callable

import numpy as np

from robustness_gym.cached_ops.spacy.spacy import Spacy
from robustness_gym.identifier import Identifier
from robustness_gym.slicemakers.subpopulations.multiplicity.multiplicity import HasMultiplicity


class HasLength(HasMultiplicity,
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
            **kwargs
        )

        # Set the intervals
        self.intervals = intervals
        self.left_limits = np.array([interval[0] for interval in intervals])
        self.right_limits = np.array([interval[1] for interval in intervals])

        # Assign the reduction fn
        self.reduction_fn = reduction_fn

    def multiplicity(self,
                     batch: Dict[str, List],
                     keys: List[str],
                     *args,
                     **kwargs) -> np.ndarray:
        # Compute the length of each example under each key
        lengths = [
            np.array([len(cache['Spacy'][key]['tokens']) for cache in batch['cache']])
            for key in keys
        ]

        # Reduction over the key axis
        return self.reduction_fn(np.array(lengths), axis=0)
