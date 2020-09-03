import json
from typing import List, Dict, Tuple, Optional

import cytoolz as tz
import numpy as np

from robustness_gym.dataset import Dataset
from robustness_gym.identifier import Identifier
from robustness_gym.slicemakers.slicemaker import SliceMaker


class Transform(SliceMaker):

    def __init__(
            self,
            category: str,
            identifiers: List[Identifier],
            apply_fn=None,
    ):
        super(Transform, self).__init__(
            category=category,
            identifiers=identifiers,
            apply_fn=apply_fn,
        )

    def apply(self,
              skeleton_batches: List[Dict[str, List]],
              slice_membership: np.ndarray,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> Tuple[List[Dict[str, List]], np.ndarray]:
        raise NotImplementedError

    def process_batch(self,
                      batch: Dict[str, List],
                      keys: List[str],
                      mask: List[int] = None,
                      store_compressed: bool = True,
                      store: bool = True) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:
        # Determine the size of the batch
        batch_size = len(batch[list(batch.keys())[0]])

        # Construct the matrix of slice labels: (batch_size x num_slices)
        slice_membership = np.ones((batch_size, self.num_slices), dtype=np.int32)

        # Uncache the batch to construct the skeleton for augmented batches
        skeleton_batches = [Dataset.uncached_batch(batch) for _ in range(self.num_slices)]

        # Set the index for the skeleton batches
        for j, skeleton_batch in enumerate(skeleton_batches):
            skeleton_batch['index'] = [f'{idx}-{self.identifiers[j]}' for idx in skeleton_batch['index']]

        # Apply the SliceMaker's core functionality
        transformed_batches, slice_membership = self.apply(
            skeleton_batches=skeleton_batches,
            slice_membership=slice_membership,
            batch=batch,
            keys=keys
        )

        # Store the transformed examples
        updates = self.construct_updates(
            transformed_batches=transformed_batches,
            slice_membership=slice_membership,
            batch_size=batch_size,
            keys=keys,
            mask=mask,
            compress=store_compressed,
        )

        # Remove transformed examples where slice_membership[i, :] = 0 before returning
        transformed_batches = [
            self.filter_batch_by_slice_membership(
                batch=transformed_batch,
                slice_membership=slice_membership[:, j:j + 1]
            )[0]
            for j, transformed_batch in enumerate(transformed_batches)
        ]

        if store:
            batch = self.store(
                batch=batch,
                updates=updates,
            )

        return batch, transformed_batches, slice_membership

    def construct_updates(self,
                          transformed_batches: List[Dict[str, List]],
                          slice_membership: np.ndarray,
                          batch_size: int,
                          keys: List[str],
                          mask: List[int] = None,
                          compress: bool = True):

        if compress:
            return [{
                        self.category: {
                            self.__class__.__name__: {
                                json.dumps(keys) if len(keys) > 1 else keys[0]: [
                                    tz.valmap(lambda v: v[i], transformed_batch)
                                    for j, transformed_batch in enumerate(transformed_batches) if slice_membership[i, j]
                                ]
                            }

                        }
                    } if np.any(slice_membership[i, :]) else {} for i in range(batch_size)]

        return [{
                    self.category: {
                        self.__class__.__name__: {
                            str(self.identifiers[j]): {
                                json.dumps(keys) if len(keys) > 1 else keys[0]: tz.valmap(lambda v: v[i],
                                                                                          transformed_batch)
                            }
                            for j, transformed_batch in enumerate(transformed_batches)
                            if (not mask or mask[j]) and (slice_membership[i, j])
                        }
                    }
                } if np.any(slice_membership[i, :]) else {} for i in range(batch_size)]
