from typing import List, Sequence

import cytoolz as tz
import numpy as np
from multiprocess.pool import Pool
from tqdm import tqdm

from robustnessgym.core.constants import SLICEBUILDERS
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.slicebuilders.subpopulation import Subpopulation


class SubpopulationCollection(Subpopulation):
    def __init__(self, subpopulations: Sequence[Subpopulation], *args, **kwargs):

        super(SubpopulationCollection, self).__init__(
            identifiers=list(
                tz.concat(
                    [subpopulation.identifiers for subpopulation in subpopulations]
                )
            ),
            *args,
            **kwargs,
        )

        # TODO(karan): some subpopulations aren't compatible with each other (e.g.
        #  single column vs. multi column):
        #  add some smarter logic here to handle this

        # Store the subpopulations
        self.subpopulations = subpopulations

    def __call__(
        self,
        dp: DataPanel,
        columns: List[str],
        num_proc: int = None,
        *args,
        **kwargs,
    ):

        if not num_proc or num_proc == 1:
            slices = []
            slice_membership = []
            # Apply each slicebuilder in sequence
            for i, slicebuilder in tqdm(enumerate(self.subpopulations)):
                # Apply the slicebuilder
                slices_i, slice_membership_i = slicebuilder(
                    dp=dp,
                    columns=columns,
                    *args,
                    **kwargs,
                )

                # Add in the slices and slice membership
                slices.extend(slices_i)
                slice_membership.append(slice_membership_i)

        else:
            # TODO(karan): cleanup, make mp.Pool support simpler across the library
            with Pool(num_proc) as pool:
                slices, slice_membership = zip(
                    *pool.map(
                        lambda sb: sb(
                            dp=dp,
                            columns=columns,
                            *args,
                            **kwargs,
                        ),
                        [slicebuilder for slicebuilder in self.subpopulations],
                    )
                )

                # Combine all the slices
                slices = list(tz.concat(slices))

            def _store_updates(batch, indices):

                # Each Subpopulation will generate slices
                for i, subpopulation in enumerate(self.subpopulations):
                    updates = subpopulation.construct_updates(
                        slice_membership=slice_membership[i][indices],
                        columns=columns,
                    )

                    batch = subpopulation.store(
                        batch=batch,
                        updates=updates,
                    )

                return batch

            if isinstance(dp, DataPanel):
                dp = dp.map(
                    _store_updates,
                    with_indices=True,
                    batched=True,
                )

                for subpopulation in self.subpopulations:
                    # Update the DataPanel's history
                    dp.update_tape(
                        path=[SLICEBUILDERS, subpopulation.category],
                        identifiers=subpopulation.identifiers,
                        columns=columns,
                    )

        # Combine all the slice membership matrices
        slice_membership = np.concatenate(slice_membership, axis=1)

        return slices, slice_membership

    def apply(
        self,
        batch: DataPanel,
        columns: List[str],
        slice_membership: np.ndarray = None,
        *args,
        **kwargs,
    ) -> np.ndarray:

        # Each Subpopulation will generate slices
        for subpopulation, end_idx in zip(
            self.subpopulations, np.cumsum([s.num_slices for s in self.subpopulations])
        ):
            # Fill out the slice_membership
            slice_membership[
                :, end_idx - subpopulation.num_slices : end_idx
            ] = subpopulation.apply(
                slice_membership=slice_membership[
                    :, end_idx - subpopulation.num_slices : end_idx
                ],
                batch=batch,
                columns=columns,
            )

        return slice_membership

    # TODO(karan): add combinations for collections
