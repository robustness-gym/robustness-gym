import json
from typing import Callable, List, Optional, Tuple

import cytoolz as tz
import numpy as np

from robustnessgym.core.constants import TRANSFORMATION
from robustnessgym.core.dataset import Batch, Dataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.slicebuilders.slicebuilder import SliceBuilder


class Transformation(SliceBuilder):
    def __init__(
        self,
        num_transformed: int = None,
        identifiers: List[Identifier] = None,
        apply_fn: Callable = None,
        category: str = None,
    ):
        assert (
            num_transformed if not identifiers else True
        ), "Must pass in num_transformed if no identifiers are given."

        super(Transformation, self).__init__(
            category=category if category else TRANSFORMATION,
            identifiers=[
                Identifier(
                    _name=f"{self.__class__.__name__}-{i + 1}",
                )
                for i in range(num_transformed)
            ]
            if not identifiers
            else identifiers,
            apply_fn=apply_fn,
        )

    @property
    def num_transformed(self):
        return self.num_slices

    def apply(
        self,
        skeleton_batches: List[Batch],
        slice_membership: np.ndarray,
        batch: Batch,
        columns: List[str],
        *args,
        **kwargs,
    ) -> Tuple[List[Batch], np.ndarray]:
        raise NotImplementedError

    def process_batch(
        self,
        batch: Batch,
        columns: List[str],
        *args,
        **kwargs,
    ) -> Tuple[List[Batch], Optional[np.ndarray]]:

        # Determine the size of the batch
        batch_size = len(batch[list(batch.keys())[0]])

        # Construct the matrix of slice labels: (batch_size x num_slices)
        slice_membership = np.ones((batch_size, self.num_slices), dtype=np.int32)

        # Uncache the batch to construct the skeleton for transformed batches
        skeleton_batches = [
            Dataset.uncached_batch(batch) for _ in range(self.num_slices)
        ]

        # Set the index for the skeleton batches
        for j, skeleton_batch in enumerate(skeleton_batches):
            skeleton_batch["index"] = [
                f"{idx}-{self.identifiers[j]}" for idx in skeleton_batch["index"]
            ]

        # Apply the SliceBuilder's core functionality
        transformed_batches, slice_membership = self.apply(
            skeleton_batches=skeleton_batches,
            slice_membership=slice_membership,
            batch=batch,
            columns=columns,
            *args,
            **kwargs,
        )

        # # Store the transformed examples
        # updates = self.construct_updates(
        #     transformed_batches=transformed_batches,
        #     slice_membership=slice_membership,
        #     batch_size=batch_size,
        #     columns=columns,
        #     mask=mask,
        #     compress=store_compressed,
        # )

        # Remove transformed examples where slice_membership[i, :] = 0 before returning
        transformed_batches = [
            self.filter_batch_by_slice_membership(
                batch=transformed_batch,
                slice_membership=slice_membership[:, j : j + 1],
            )[0]
            for j, transformed_batch in enumerate(transformed_batches)
        ]

        # if store:
        #     batch = self.store(
        #         batch=batch,
        #         updates=updates,
        #     )

        return transformed_batches, slice_membership

    def construct_updates(
        self,
        transformed_batches: List[Batch],
        slice_membership: np.ndarray,
        batch_size: int,
        columns: List[str],
        mask: List[int] = None,
        compress: bool = True,
    ):

        if compress:
            return [
                {
                    self.category: {
                        self.__class__.__name__: {
                            json.dumps(columns)
                            if len(columns) > 1
                            else columns[0]: [
                                tz.valmap(lambda v: v[i], transformed_batch)
                                for j, transformed_batch in enumerate(
                                    transformed_batches
                                )
                                if slice_membership[i, j]
                            ]
                        }
                    }
                }
                if np.any(slice_membership[i, :])
                else {}
                for i in range(batch_size)
            ]

        return [
            {
                self.category: {
                    self.__class__.__name__: {
                        str(self.identifiers[j]): {
                            json.dumps(columns)
                            if len(columns) > 1
                            else columns[0]: tz.valmap(
                                lambda v: v[i], transformed_batch
                            )
                        }
                        for j, transformed_batch in enumerate(transformed_batches)
                        if (not mask or not mask[j]) and (slice_membership[i, j])
                    }
                }
            }
            if np.any(slice_membership[i, :])
            else {}
            for i in range(batch_size)
        ]


class SingleColumnTransformation(Transformation):
    def single_column_apply(self, column_batch: List) -> List[List]:
        raise NotImplementedError

    def apply(
        self,
        skeleton_batches: List[Batch],
        slice_membership: np.ndarray,
        batch: Batch,
        columns: List[str],
        *args,
        **kwargs,
    ) -> Tuple[List[Batch], np.ndarray]:

        # Independently apply the transformation over the columns
        for column in columns:
            try:
                # Apply
                transformed_batch = self.single_column_apply(
                    column_batch=batch[column],
                )

                assert len(transformed_batch) == len(
                    batch[column]
                ), "Must output one list of augmentations per example."

                # Store the transformed text in the skeleton batches
                for i in range(slice_membership.shape[0]):
                    for j, transformed in enumerate(transformed_batch[i]):
                        skeleton_batches[j][column][i] = transformed
            except:  # noqa
                # Unable to transform: set all slice membership labels to zero
                slice_membership[:, :] = 0

        return skeleton_batches, slice_membership
