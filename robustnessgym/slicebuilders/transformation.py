from typing import Callable, List, Optional, Tuple

import numpy as np

from robustnessgym.core.constants import TRANSFORMATION
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.slice import SliceDataPanel as DataPanel
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
            identifiers=[
                Identifier(
                    _name=f"{self.__class__.__name__}-{i + 1}",
                )
                for i in range(num_transformed)
            ]
            if not identifiers
            else identifiers,
            category=category if category else TRANSFORMATION,
            apply_fn=apply_fn,
        )

    @property
    def num_transformed(self):
        return self.num_slices

    def apply(
        self,
        batch: DataPanel,
        columns: List[str],
        skeleton_batches: List[DataPanel],
        slice_membership: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[List[DataPanel], np.ndarray]:
        raise NotImplementedError

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        *args,
        **kwargs,
    ) -> Tuple[List[DataPanel], Optional[np.ndarray]]:

        # Determine the size of the batch
        batch_size = len(dp[list(dp.keys())[0]])

        # Construct the matrix of slice labels: (batch_size x num_slices)
        slice_membership = np.ones((batch_size, self.num_slices), dtype=np.int32)

        # Uncache the batch to construct the skeleton for transformed batches
        skeleton_batches = [
            DataPanel.uncached_batch(dp) for _ in range(self.num_slices)
        ]

        # Set the index for the skeleton batches
        for j, skeleton_batch in enumerate(skeleton_batches):
            # skeleton_batch.update(
            #     lambda x: {'index': f"{x['index']}-{self.identifiers[j]}"}
            # )
            skeleton_batch["index"] = [
                f"{idx}-{self.identifiers[j]}" for idx in skeleton_batch["index"]
            ]

        # Apply the SliceBuilder's core functionality: use positional args
        try:
            transformed_batches, slice_membership = self.apply(
                dp,
                columns,
                skeleton_batches,
                slice_membership,
                *args,
                **kwargs,
            )
        except TypeError:
            self.apply(dp, columns, *args, **kwargs)

        # Remove transformed examples where slice_membership[i, :] = 0 before returning
        transformed_batches = [
            self.filter_batch_by_slice_membership(
                batch=transformed_batch,
                slice_membership=slice_membership[:, j : j + 1],
            )[0]
            for j, transformed_batch in enumerate(transformed_batches)
        ]

        return transformed_batches, slice_membership


class SingleColumnTransformation(Transformation):
    def single_column_apply(self, column_batch: List) -> List[List]:
        return NotImplemented(
            "Implement single_column_apply to use this transformation."
        )

    def apply(
        self,
        batch: DataPanel,
        columns: List[str],
        skeleton_batches: List[DataPanel],
        slice_membership: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[List[DataPanel], np.ndarray]:

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
