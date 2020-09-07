from __future__ import annotations

from typing import List, Dict

import numpy as np

from robustness_gym.cached_ops.spacy.spacy import Spacy
from robustness_gym.dataset import Dataset
from robustness_gym.identifier import Identifier
from robustness_gym.slicemakers.subpopulation import Subpopulation


class MinLength(Subpopulation,
                Spacy):

    def __init__(self,
                 thresholds: List[int],
                 *args,
                 **kwargs):

        super(MinLength, self).__init__(
            # One slice per length threshold
            identifiers=[
                Identifier(name=self.__class__.__name__,
                           threshold=threshold)
                for threshold in thresholds
            ],
            *args,
            **kwargs
        )

        # This is the list of min cutoff values for length, one for each slice
        self.thresholds = thresholds

    @classmethod
    def from_dataset(cls, dataset: Dataset, precentiles: List[float]) -> MinLength:
        """
        Determine thresholds from percentiles for specific dataset
        """
        raise NotImplementedError

    def apply(self,
              slice_membership: np.ndarray,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> np.ndarray:

        if len(keys) != 1:
            raise ValueError('Only one key allowed')

        # Use the spacy cache to grab the tokens in each example (for each key)
        tokenized_batch = self.get_tokens(batch, keys)

        # Check that number of tokens is equal to or greater than min length
        for i, example in enumerate(tokenized_batch):
            tokens = example[keys[0]]
            for j, threshold in enumerate(self.thresholds):
                if len(tokens) >= threshold:
                    slice_membership[i, j] = 1

        return slice_membership
