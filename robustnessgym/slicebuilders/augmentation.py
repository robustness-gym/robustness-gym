from typing import List, Callable, Dict, Tuple

import numpy as np

from robustnessgym.constants import AUGMENTATION
from robustnessgym.dataset import Batch
from robustnessgym.identifier import Identifier
from robustnessgym.slicebuilders.transform import Transform


class Augmentation(Transform):

    def __init__(
            self,
            num_transformed: int = None,
            identifiers: List[Identifier] = None,
            apply_fn: Callable = None,
    ):
        assert num_transformed if not identifiers else True, \
            "Must pass in num_transformed if no identifiers are given."

        super(Augmentation, self).__init__(
            category=AUGMENTATION,
            identifiers=[
                Identifier(
                    _name=f"{self.__class__.__name__}-{i + 1}",
                )
                for i in range(num_transformed)
            ] if not identifiers else identifiers,
            apply_fn=apply_fn,
        )


class SingleColumnAugmentation(Augmentation):

    def single_column_apply(self,
                            column_batch: List) -> List[List]:
        raise NotImplementedError

    def apply(self,
              skeleton_batches: List[Batch],
              slice_membership: np.ndarray,
              batch: Batch,
              columns: List[str],
              *args,
              **kwargs) -> Tuple[List[Batch], np.ndarray]:

        # Independently apply the augmentation over the columns
        for column in columns:
            try:
                # Apply
                augmented_batch = self.single_column_apply(
                    column_batch=batch[column],
                )

                assert len(augmented_batch) == len(batch[column]), \
                    "Must output one list of augmentations per example."

                # Store the augmented text in the skeleton batches
                for i in range(slice_membership.shape[0]):
                    for j, augmented in enumerate(augmented_batch[i]):
                        skeleton_batches[j][column][i] = augmented
            except:
                # Unable to augment: set all slice membership labels to zero
                slice_membership[:, :] = 0

        return skeleton_batches, slice_membership
