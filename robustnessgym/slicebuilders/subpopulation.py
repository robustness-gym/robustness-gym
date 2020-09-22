import json
from typing import List, Dict, Tuple, Optional

import cytoolz as tz
import numpy as np

from robustnessgym.constants import *
from robustnessgym.identifier import Identifier
from robustnessgym.slicebuilders.slicebuilder import SliceBuilder


class Subpopulation(SliceBuilder):

    def __init__(
            self,
            identifiers: List[Identifier],
            apply_fn=None,
            *args,
            **kwargs
    ):
        super(Subpopulation, self).__init__(
            category=SUBPOPULATION,
            identifiers=identifiers,
            apply_fn=apply_fn,
            *args,
            **kwargs,
        )

    def apply(self,
              slice_membership: np.ndarray,
              batch: Dict[str, List],
              columns: List[str],
              *args,
              **kwargs) -> np.ndarray:
        raise NotImplementedError

    def process_batch(self,
                      batch: Dict[str, List],
                      columns: List[str],
                      mask: List[int] = None,
                      store_compressed: bool = True,
                      store: bool = True,
                      *args,
                      **kwargs) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:

        # Determine the size of the batch
        batch_size = len(batch[list(batch.keys())[0]])

        # Construct the matrix of slice labels: (batch_size x num_slices)
        slice_membership = np.zeros((batch_size, self.num_slices), dtype=np.int32)

        # Apply the SliceMaker's core functionality
        slice_membership = self.apply(slice_membership, batch, columns, *args, **kwargs)

        # Store these slice labels
        # TODO(karan): figure out how to set the alias
        updates = self.construct_updates(
            slice_membership=slice_membership,
            columns=columns,
            mask=mask,
            compress=store_compressed
        )

        if store:
            batch = self.store(
                batch=batch,
                updates=updates,
            )

        return batch, self.filter_batch_by_slice_membership(batch, slice_membership), slice_membership

    def construct_updates(self,
                          slice_membership: np.ndarray,
                          columns: List[str],
                          mask: List[int] = None,
                          compress: bool = True):

        # Mask out components
        # TODO(karan): masking inside apply, but only if the components are computed independently

        # Construct a list of update dicts that contains the slice membership for each example
        if compress:
            # TODO(karan): this will overwrite a previous application of the same Slicer right now, need a merge operation
            # Merge is just an append to whatever list already exists
            return [{
                self.category: {
                    self.__class__.__name__: {
                        json.dumps(columns) if len(columns) > 1 else columns[0]: row
                    }
                }
            } for row in (slice_membership[:, np.logical_not(np.array(mask, dtype=bool))]
                          if mask else slice_membership).tolist()]

        return [{
            self.category: {
                self.__class__.__name__: {
                    str(self.identifiers[i]): {
                        json.dumps(columns) if len(columns) > 1 else columns[0]: membership
                    }
                    for i, membership in enumerate(row) if not mask or not mask[i]
                },
            }
        } for row in slice_membership.tolist()]

    @classmethod
    def union(cls,
              *slicemakers: SliceBuilder,
              identifier: Identifier = None) -> SliceBuilder:
        """
        Combine a list of slicers using a union.
        """
        # Group the slicers based on their class
        grouped_slicers = tz.groupby(lambda s: s.__class__, slicemakers)

        # Join the slicers corresponding to each class, and flatten
        slicemakers = list(tz.concat(
            tz.itemmap(lambda item: (item[0], item[0].join(*item[1])),
                       grouped_slicers).values())
        )

        def apply_fn(slice_membership,
                     batch,
                     columns,
                     *args,
                     **kwargs):
            # Determine the size of the batch
            batch_size = len(batch[list(batch.keys())[0]])

            # Keep track of all the slice labels
            all_slice_membership = []

            # Run each slicemaker on the batch
            for slicemaker in slicemakers:
                all_slice_membership.append(
                    slicemaker.apply(
                        slice_membership=np.zeros((batch_size, slicemaker.num_slices), dtype=np.int32),
                        batch=batch,
                        columns=columns
                    )
                )

            # Concatenate all the slice labels
            slice_membership = np.concatenate(all_slice_membership, axis=1)

            # Take the union over the slices (columns)
            slice_membership = np.any(slice_membership, axis=1).astype(np.int32)[:, np.newaxis]

            return slice_membership

        return Subpopulation(identifiers=[identifier], apply_fn=apply_fn)

    @classmethod
    def intersection(cls,
                     *slicemakers: SliceBuilder,
                     identifier: Identifier = None) -> SliceBuilder:
        """
        Combine a list of slicemakers using an intersection.
        """
        # Group the slicemakers based on their class
        grouped_slicemakers = tz.groupby(lambda s: s.__class__, slicemakers)

        # Join the slicemakers corresponding to each class, and flatten
        slicemakers = list(tz.concat(
            tz.itemmap(lambda item: (item[0], item[0].join(*item[1])),
                       grouped_slicemakers).values())
        )

        def apply_fn(slice_membership,
                     batch,
                     columns,
                     *args,
                     **kwargs):
            # Determine the size of the batch
            batch_size = len(batch[list(batch.keys())[0]])

            # Keep track of all the slice labels
            all_slice_membership = []

            # Run each slicemaker on the batch
            for slicemaker in slicemakers:
                all_slice_membership.append(
                    slicemaker.apply(
                        slice_membership=np.zeros((batch_size, slicemaker.num_slices), dtype=np.int32),
                        batch=batch,
                        columns=columns
                    )
                )

            # Concatenate all the slice labels
            slice_membership = np.concatenate(all_slice_membership, axis=1)

            # Take the union over the slices (columns)
            slice_membership = np.all(slice_membership, axis=1).astype(np.int32)[:, np.newaxis]

            return slice_membership

        return Subpopulation(identifiers=[identifier], apply_fn=apply_fn)


class SubpopulationCollection(Subpopulation):

    def __init__(self,
                 subpopulations: List[Subpopulation],
                 *args,
                 **kwargs):
        super(SubpopulationCollection, self).__init__(
            identifiers=list(tz.concat([subpopulation.identifiers for subpopulation in subpopulations])),
        )

        # Store the subpopulations
        self.subpopulations = subpopulations

    def apply(self,
              slice_membership: np.ndarray,
              batch: Dict[str, List],
              columns: List[str],
              *args,
              **kwargs) -> np.ndarray:
        # Each Subpopulation will generate slices
        for subpopulation, end_idx in zip(self.subpopulations, np.cumsum([s.num_slices for s in self.subpopulations])):
            # Fill out the slice_membership
            slice_membership[:, end_idx - subpopulation.num_slices:end_idx] = subpopulation.apply(
                slice_membership=slice_membership[:, end_idx - subpopulation.num_slices:end_idx],
                batch=batch,
                columns=columns
            )

        return slice_membership
