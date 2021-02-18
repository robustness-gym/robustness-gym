import json
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cytoolz as tz
import numpy as np
from multiprocess.pool import Pool
from tqdm import tqdm

from robustnessgym.core.constants import SLICEBUILDERS, SUBPOPULATION
from robustnessgym.core.dataset import Batch, Dataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.slice import Slice
from robustnessgym.core.tools import recmerge, strings_as_json
from robustnessgym.slicebuilders.slicebuilder import SliceBuilder


class Subpopulation(SliceBuilder):
    def __init__(self, identifiers: List[Identifier], apply_fn=None, *args, **kwargs):
        super(Subpopulation, self).__init__(
            category=SUBPOPULATION,
            identifiers=identifiers,
            apply_fn=apply_fn,
            *args,
            **kwargs,
        )

    def apply(
        self,
        slice_membership: np.ndarray,
        batch: Batch,
        columns: List[str],
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def process_batch(
        self,
        batch: Dict[str, List],
        columns: List[str],
        # mask: List[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[Optional[List[Batch]], Optional[np.ndarray]]:

        # Determine the size of the batch
        batch_size = len(batch[list(batch.keys())[0]])

        # Construct the matrix of slice labels: (batch_size x num_slices)
        slice_membership = np.zeros((batch_size, self.num_slices), dtype=np.int32)

        # Apply the SliceBuilder's core functionality
        slice_membership = self.apply(slice_membership, batch, columns, *args, **kwargs)

        # # Store these slice labels
        # # TODO(karan): figure out how to set the alias
        # updates = self.construct_updates(
        #     slice_membership=slice_membership,
        #     columns=columns,
        #     mask=mask,
        #     compress=store_compressed,
        # )
        #
        # if store:
        #     batch = self.store(
        #         batch=batch,
        #         updates=updates,
        #     )

        return (
            None,  # self.filter_batch_by_slice_membership(batch, slice_membership),
            slice_membership,
        )

    def process_dataset(
        self,
        dataset: Dataset,
        columns: List[str],
        batch_size: int = 32,
        # mask: List[int] = None,
        num_proc: int = None,
        *args,
        **kwargs,
    ) -> Tuple[List[Slice], np.ndarray]:

        # Create slices using the dataset
        slices = [Slice(dataset) for _ in range(len(self.identifiers))]
        all_slice_memberships = []
        # Batch the dataset, and process each batch
        for batch in dataset.batch(batch_size):
            # Process the batch
            _, slice_memberships = self.process_batch(
                batch=batch,
                columns=columns,
                *args,
                **kwargs,
            )

            # Keep track of the slice memberships
            all_slice_memberships.append(slice_memberships)

        # Create a single slice label matrix
        slice_membership = np.concatenate(all_slice_memberships, axis=0)

        for i, sl in enumerate(slices):
            # Set the visible rows for each slice
            sl.set_visible_rows(np.where(slice_membership[:, i])[0])

            # Set the Slice category using the SliceBuilder's category
            sl.category = self.category

            # Append the the lineage
            sl.add_to_lineage(
                category=str(self.category.capitalize()),
                identifier=self.identifiers[i],
                columns=strings_as_json(columns),
            )

            # # Create the lineage
            # sl.lineage = [
            #     (str(Dataset.__name__), dataset.identifier),
            #     (
            #         str(self.category.capitalize()),
            #         self.identifiers[i],
            #         strings_as_json(columns),
            #     ),
            # ]
            # if isinstance(dataset, Slice):
            #     # Prepend the Slice's lineage instead, if the dataset was a slice
            #     sl.lineage = dataset.lineage + [
            #         (
            #             str(self.category.capitalize()),
            #             self.identifiers[i],
            #             strings_as_json(columns),
            #         )
            #     ]

        return slices, slice_membership

    def construct_updates(
        self,
        slice_membership: np.ndarray,
        columns: List[str],
        mask: List[int] = None,
        compress: bool = True,
    ):

        # Mask out components
        # TODO(karan): masking inside apply, but only if the components are computed
        #  independently

        # Construct a list of update dicts that contains the slice membership for
        # each example
        if compress:
            # TODO(karan): this will overwrite a previous application of the same
            #  Slicer right now, need a merge operation
            # Merge is just an append to whatever list already exists
            return [
                {
                    self.category: {
                        self.__class__.__name__: {
                            json.dumps(columns) if len(columns) > 1 else columns[0]: row
                        }
                    }
                }
                for row in (
                    slice_membership[:, np.logical_not(np.array(mask, dtype=bool))]
                    if mask
                    else slice_membership
                ).tolist()
            ]

        return [
            {
                self.category: {
                    self.__class__.__name__: {
                        str(self.identifiers[i]): {
                            json.dumps(columns)
                            if len(columns) > 1
                            else columns[0]: membership
                        }
                        for i, membership in enumerate(row)
                        if not mask or not mask[i]
                    },
                }
            }
            for row in slice_membership.tolist()
        ]

    @classmethod
    def union(
        cls, *slicemakers: SliceBuilder, identifier: Identifier = None
    ) -> SliceBuilder:
        """Combine a list of slicers using a union."""
        # Group the slicers based on their class
        grouped_slicers = tz.groupby(lambda s: s.__class__, slicemakers)

        # Join the slicers corresponding to each class, and flatten
        slicemakers = list(
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

            # Run each slicemaker on the batch
            for slicemaker in slicemakers:
                all_slice_membership.append(
                    slicemaker.apply(
                        slice_membership=np.zeros(
                            (batch_size, slicemaker.num_slices), dtype=np.int32
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
        cls, *slicemakers: SliceBuilder, identifier: Identifier = None
    ) -> SliceBuilder:
        """Combine a list of slicemakers using an intersection."""
        # Group the slicemakers based on their class
        grouped_slicemakers = tz.groupby(lambda s: s.__class__, slicemakers)

        # Join the slicemakers corresponding to each class, and flatten
        slicemakers = list(
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
            for slicemaker in slicemakers:
                all_slice_membership.append(
                    slicemaker.apply(
                        slice_membership=np.zeros(
                            (batch_size, slicemaker.num_slices), dtype=np.int32
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
        batch_or_dataset: Union[Batch, Dataset],
        columns: List[str],
        mask: List[int] = None,
        store_compressed: bool = None,
        store: bool = None,
        num_proc: int = None,
        *args,
        **kwargs,
    ):

        if mask:
            raise NotImplementedError(
                "Mask not supported for SubpopulationCollection yet."
            )

        if not num_proc or num_proc == 1:
            slices = []
            slice_membership = []
            # Apply each slicebuilder in sequence
            for i, slicebuilder in tqdm(enumerate(self.subpopulations)):
                # Apply the slicebuilder
                batch_or_dataset, slices_i, slice_membership_i = slicebuilder(
                    batch_or_dataset=batch_or_dataset,
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

        else:
            # TODO(karan): cleanup, make mp.Pool support simpler across the library
            with Pool(num_proc) as pool:
                batches_or_datasets, slices, slice_membership = zip(
                    *pool.map(
                        lambda sb: sb(
                            batch_or_dataset=batch_or_dataset,
                            columns=columns,
                            mask=mask,
                            store_compressed=store_compressed,
                            store=store,
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
                        mask=mask,
                        # TODO(karan): this option should be set correctly
                        compress=True,
                    )

                    batch = subpopulation.store(
                        batch=batch,
                        updates=updates,
                    )

                return batch

            if isinstance(batch_or_dataset, Dataset):
                batch_or_dataset = batch_or_dataset.map(
                    _store_updates,
                    with_indices=True,
                    batched=True,
                )

                for subpopulation in self.subpopulations:
                    # Update the Dataset's history
                    batch_or_dataset.update_tape(
                        path=[SLICEBUILDERS, subpopulation.category],
                        identifiers=subpopulation.identifiers,
                        columns=columns,
                    )

            else:
                batch_or_dataset = recmerge(*batches_or_datasets, merge_sequences=True)

        # Combine all the slice membership matrices
        slice_membership = np.concatenate(slice_membership, axis=1)

        return batch_or_dataset, slices, slice_membership

    def apply(
        self,
        slice_membership: np.ndarray,
        batch: Batch,
        columns: List[str],
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
