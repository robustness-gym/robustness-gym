from typing import List, Optional, Tuple

import cytoolz as tz
import numpy as np
from meerkat.provenance import capture_provenance

from robustnessgym.core.constants import SUBPOPULATION
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.core.tools import strings_as_json
from robustnessgym.slicebuilders.slicebuilder import SliceBuilder


class Subpopulation(SliceBuilder):
    def __init__(self, identifiers: List[Identifier], apply_fn=None, *args, **kwargs):
        super(Subpopulation, self).__init__(
            identifiers=identifiers,
            category=SUBPOPULATION,
            apply_fn=apply_fn,
            *args,
            **kwargs,
        )

    def apply(
        self,
        batch: DataPanel,
        columns: List[str],
        slice_membership: np.ndarray = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return NotImplemented("Must return a np.ndarray matrix.")

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        *args,
        **kwargs,
    ) -> Tuple[Optional[List[DataPanel]], Optional[np.ndarray]]:

        # Construct the matrix of slice labels: (batch_size x num_slices)
        slice_membership = np.zeros((len(dp), self.num_slices), dtype=np.int32)

        # Apply the SliceBuilder's core functionality
        try:
            slice_membership = self.apply(
                dp, columns, slice_membership, *args, **kwargs
            )
        except TypeError:
            slice_membership = np.array(
                self.apply(dp, columns, *args, **kwargs)
            ).reshape((-1, self.num_slices))

        assert slice_membership.shape == (len(dp), self.num_slices), (
            "Output of apply has the wrong shape. "
            f"Expected: {(len(dp), self.num_slices)} and got: {slice_membership.shape}."
        )

        return None, slice_membership

    @capture_provenance(capture_args=["self", "columns", "batch_size"])
    def process_dataset(
        self,
        dp: DataPanel,
        columns: List[str],
        batch_size: int = 32,
        num_proc: int = None,
        *args,
        **kwargs,
    ) -> Tuple[List[DataPanel], np.ndarray]:

        # Create slices using the dataset
        all_slice_memberships = []

        # Batch the dataset, and process each batch
        for batch in dp.batch(batch_size):
            # Process the batch
            _, slice_memberships = self.process_batch(
                dp=batch,
                columns=columns,
                *args,
                **kwargs,
            )

            # Keep track of the slice memberships
            all_slice_memberships.append(slice_memberships)

        # Create a single slice label matrix
        slice_membership = np.concatenate(all_slice_memberships, axis=0)

        slices = []
        for i in range(len(self.identifiers)):
            # Create a view of the original DataPanel
            sl = dp.view()

            # Only keep the filtered rows visible
            for column in sl._data.values():
                column.visible_rows = np.where(slice_membership[:, i])[0]

            # Set the Slice category using the SliceBuilder's category
            sl.category = self.category

            # Append the the lineage
            sl.add_to_lineage(
                category=str(self.category.capitalize()),
                identifier=self.identifiers[i],
                columns=strings_as_json(columns),
            )

            #
            # sl.identifier = ...

            slices.append(sl)

        # for i, sl in enumerate(slices):
        #     # Set the visible rows for each slice
        #     sl.set_visible_rows(np.where(slice_membership[:, i])[0])

        return slices, slice_membership

    @classmethod
    def union(
        cls, *slicebuilders: SliceBuilder, identifier: Identifier = None
    ) -> SliceBuilder:
        """Combine a list of slicers using a union."""
        # Group the slicers based on their class
        grouped_slicers = tz.groupby(lambda s: s.__class__, slicebuilders)

        # Join the slicers corresponding to each class, and flatten
        slicebuilders = list(
            tz.concat(
                tz.itemmap(
                    lambda item: (item[0], item[0].join(*item[1])), grouped_slicers
                ).values()
            )
        )

        def apply_fn(slice_membership, batch, columns, *args, **kwargs):
            # Determine the size of the batch
            batch_size = len(batch[list(batch.keys())[0]])

            # Keep track of all the slice labels
            all_slice_membership = []

            # Run each slicebuilder on the batch
            for slicebuilder in slicebuilders:
                all_slice_membership.append(
                    slicebuilder.apply(
                        slice_membership=np.zeros(
                            (batch_size, slicebuilder.num_slices), dtype=np.int32
                        ),
                        batch=batch,
                        columns=columns,
                    )
                )

            # Concatenate all the slice labels
            slice_membership = np.concatenate(all_slice_membership, axis=1)

            # Take the union over the slices (columns)
            slice_membership = np.any(slice_membership, axis=1).astype(np.int32)[
                :, np.newaxis
            ]

            return slice_membership

        return Subpopulation(identifiers=[identifier], apply_fn=apply_fn)

    @classmethod
    def intersection(
        cls, *slicebuilders: SliceBuilder, identifier: Identifier = None
    ) -> SliceBuilder:
        """Combine a list of slicemakers using an intersection."""
        # Group the slicemakers based on their class
        grouped_slicemakers = tz.groupby(lambda s: s.__class__, slicebuilders)

        # Join the slicemakers corresponding to each class, and flatten
        slicebuilders = list(
            tz.concat(
                tz.itemmap(
                    lambda item: (item[0], item[0].join(*item[1])), grouped_slicemakers
                ).values()
            )
        )

        def apply_fn(slice_membership, batch, columns, *args, **kwargs):
            # Determine the size of the batch
            batch_size = len(batch[list(batch.keys())[0]])

            # Keep track of all the slice labels
            all_slice_membership = []

            # Run each slicemaker on the batch
            for slicebuilder in slicebuilders:
                all_slice_membership.append(
                    slicebuilder.apply(
                        slice_membership=np.zeros(
                            (batch_size, slicebuilder.num_slices), dtype=np.int32
                        ),
                        batch=batch,
                        columns=columns,
                    )
                )

            # Concatenate all the slice labels
            slice_membership = np.concatenate(all_slice_membership, axis=1)

            # Take the union over the slices (columns)
            slice_membership = np.all(slice_membership, axis=1).astype(np.int32)[
                :, np.newaxis
            ]

            return slice_membership

        return Subpopulation(identifiers=[identifier], apply_fn=apply_fn)
