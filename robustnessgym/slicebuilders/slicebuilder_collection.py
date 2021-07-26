from __future__ import annotations

from typing import List

import cytoolz as tz
import numpy as np
import tqdm

from robustnessgym.core.constants import GENERIC
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.slicebuilders.slicebuilder import SliceBuilder


class SliceBuilderCollection(SliceBuilder):
    """Collection of Slice Builders."""

    def __init__(self, slicebuilders: List[SliceBuilder], *args, **kwargs):
        super(SliceBuilderCollection, self).__init__(
            category=GENERIC,
            identifiers=list(
                tz.concat([slicebuilder.identifiers for slicebuilder in slicebuilders])
            ),
            *args,
            **kwargs,
        )

        # TODO(karan): some slicebuilders aren't compatible with each other (e.g.
        #  single column vs. multi column):
        #  add some smarter logic here to handle this

        # Store the subpopulations
        self.slicebuilders = slicebuilders

    def __repr__(self):
        # TODO(karan): format this nicely
        return (
            f"{self.__class__.__name__}("
            f"{[str(slicebuilder) for slicebuilder in self.slicebuilders]})]"
        )

    def __call__(
        self,
        dp: DataPanel,
        columns: List[str],
        mask: List[int] = None,
        store_compressed: bool = None,
        store: bool = None,
        *args,
        **kwargs,
    ):

        if mask:
            raise NotImplementedError(
                "Mask not supported for SliceBuilderCollection yet."
            )

        slices = []
        slice_membership = []

        # Apply each slicebuilder in sequence
        for i, slicebuilder in tqdm.tqdm(enumerate(self.slicebuilders)):
            # Apply the slicebuilder
            dp, slices_i, slice_membership_i = slicebuilder(
                batch_or_dataset=dp,
                columns=columns,
                mask=mask,
                store_compressed=store_compressed,
                store=store,
                *args,
                **kwargs,
            )

            # Add in the slices and slice membership
            slices.extend(slices_i)
            slice_membership.append(slice_membership_i)

        slice_membership = np.concatenate(slice_membership, axis=1)

        return dp, slices, slice_membership
