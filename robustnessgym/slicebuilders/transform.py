import json
from typing import List, Dict, Tuple, Optional

import cytoolz as tz
import numpy as np

from robustnessgym.dataset import Dataset, Batch
from robustnessgym.identifier import Identifier
from robustnessgym.slicebuilders.slicebuilder import SliceBuilder


class Transform(SliceBuilder):

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

    @property
    def num_transformed(self):
        return self.num_slices

    def apply(self,
              skeleton_batches: List[Batch],
              slice_membership: np.ndarray,
              batch: Batch,
              columns: List[str],
              *args,
              **kwargs) -> Tuple[List[Batch], np.ndarray]:
        raise NotImplementedError

    def process_batch(self,
                      batch: Batch,
                      columns: List[str],
                      mask: List[int] = None,
                      store_compressed: bool = True,
                      store: bool = True,
                      *args,
                      **kwargs) -> Tuple[Batch, List[Batch], Optional[np.ndarray]]:
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
            columns=columns,
            *args,
            **kwargs,
        )

        # Store the transformed examples
        updates = self.construct_updates(
            transformed_batches=transformed_batches,
            slice_membership=slice_membership,
            batch_size=batch_size,
            columns=columns,
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
                          transformed_batches: List[Batch],
                          slice_membership: np.ndarray,
                          batch_size: int,
                          columns: List[str],
                          mask: List[int] = None,
                          compress: bool = True):

        if compress:
            return [{
                        self.category: {
                            self.__class__.__name__: {
                                json.dumps(columns) if len(columns) > 1 else columns[0]: [
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
                                json.dumps(columns) if len(columns) > 1 else columns[0]: tz.valmap(lambda v: v[i],
                                                                                                   transformed_batch)
                            }
                            for j, transformed_batch in enumerate(transformed_batches)
                            if (not mask or not mask[j]) and (slice_membership[i, j])
                        }
                    }
                } if np.any(slice_membership[i, :]) else {} for i in range(batch_size)]
