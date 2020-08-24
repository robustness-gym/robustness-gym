from __future__ import annotations

from abc import ABCMeta, ABC
from functools import partial
from itertools import compress
from typing import *

import spacy
import cytoolz as tz
from pyarrow import json
import numpy as np
import nlp
import streamlit as st
import torch
from allennlp.predictors import Predictor
import allennlp_models.structured_prediction.predictors.constituency_parser
from quinine.common.utils import rmerge
from robustness_gym.dataset import CachedOperation, Dataset
from robustness_gym.slice import Slice
import robustness_gym
import dill as pickle


class PicklerMixin:

    def __init__(self, *args, **kwargs):
        pass

    def save(self, path: str) -> None:
        """
        Save the object.
        """
        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def load(cls, path: str):
        """
        Load the object from the path.
        """
        return pickle.load(open(path, 'rb'))


SUBPOPULATION = 'subpopulation'
ATTACK = 'attack'
AUGMENTATION = 'augmentation'
CURATION = 'curation'


class SliceMaker(PicklerMixin):
    CATEGORIES = [
        SUBPOPULATION,
        ATTACK,
        AUGMENTATION,
        CURATION,
    ]

    def __init__(self,
                 category: str,
                 num_slices: int,
                 identifiers: List[str],
                 apply_fn: Callable = None,
                 *args,
                 **kwargs):
        super(SliceMaker, self).__init__(*args, **kwargs)

        assert category in self.CATEGORIES, f"argument category must be one of {self.CATEGORIES}"
        self.category = category

        self.tasks = None

        self.metadata = {}

        self.num_slices = num_slices
        self.identifiers = identifiers

        assert len(self.identifiers) == self.num_slices, "Must have exactly one identifier per slice."

        # Keep track of the CachedOperation dependencies
        self.prerequisites = set()
        for base in self.__class__.__bases__:
            if isinstance(base, CachedOperation):
                self.prerequisites.add(base.__name__)

        if apply_fn:
            # Assign to the method
            self.apply = apply_fn

    def __call__(self, batch_or_dataset, keys):

        if isinstance(batch_or_dataset, Dataset):
            # Slice a dataset
            return self.process_dataset(dataset=batch_or_dataset,
                                        keys=keys)
        elif isinstance(batch_or_dataset, Dict):
            # Slice a batch
            return self.process_batch(batch=batch_or_dataset,
                                      keys=keys)
        else:
            raise NotImplementedError

    def alias(self) -> str:
        return self.__class__.__name__

    def process_dataset(self,
                        dataset: Dataset,
                        keys: List[str],
                        batch_size: int = 32) -> Tuple[Dataset, List[Slice], np.ndarray]:
        """
        Slice a dataset.
        """

        # Batch the dataset, and slice each batch
        all_batches, all_sliced_batches, all_slice_labels = zip(
            *[self.process_batch(tz.merge_with(tz.identity, *batch), keys)
              for batch in tz.partition_all(batch_size, dataset)]
        )

        # TODO(karan): want to do this instead but .map() in Huggingface nlp must return either a None type or dict
        # all_batches, all_sliced_batches, all_slice_labels = \
        #     zip(*dataset.map(lambda examples: self.slice_batch(batch=examples, keys=keys),
        #                      batched=True, batch_size=batch_size))

        # Update the dataset efficiently by reusing all_batches
        # TODO(karan): have to run this separately since .map() only updates the dataset if a dict is returned
        dataset = dataset.map(lambda examples, indices: all_batches[indices[0] // batch_size],
                              batched=True, batch_size=batch_size, with_indices=True)

        # Create the dataset slices
        slices = [Slice.from_batches(slice_batches)
                  for slice_batches in zip(*all_sliced_batches)]

        # Create a single slice label matrix
        slice_labels = np.concatenate(all_slice_labels, axis=0)

        return dataset, slices, slice_labels

    def process_batch(self,
                      batch: Dict[str, List],
                      keys: List[str]) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:
        return batch, [batch], None

    def apply(self, *args, **kwargs):
        raise NotImplementedError("Must implement apply.")

    @classmethod
    def join(cls, *slicemakers: SliceMaker) -> Sequence[SliceMaker]:
        """
        Join many slicemakers. By default, just returns the slicemakers.
        """
        return slicemakers


# Example stores
# binary vector of slice membership

class Augmentation(SliceMaker):

    def __init__(
            self,
            num_slices,
            identifiers,
            slice_batch_fn,
    ):
        super(Augmentation, self).__init__(
            category=AUGMENTATION,
            num_slices=num_slices,
            identifiers=identifiers,
            slice_batch_fn=slice_batch_fn,
        )

    @staticmethod
    def store(batch: Dict[str, List],
              augmented_batches: List[Dict[str, List]],
              key: str):
        """
        Update a batch of examples with augmented examples.
        """
        batch['slices'] = [rmerge(example_dict,
                                  {AUGMENTATION: {key: [tz.valmap(lambda v: v[i], aug_batch)
                                                        for aug_batch in augmented_batches]}})
                           for i, example_dict in enumerate(batch['slices'])]
        return batch


class Attack(SliceMaker):

    def __init__(
            self,
            num_slices,
            identifiers,
            slice_batch_fn=None,
    ):
        super(Attack, self).__init__(
            category=ATTACK,
            num_slices=num_slices,
            identifiers=identifiers,
            slice_batch_fn=slice_batch_fn,
        )

    def apply(self, batch: Dict[str, List], keys: List[str]):
        pass

    @staticmethod
    def store(batch: Dict[str, List],
              attacked_batches: List[Dict[str, List]],
              key: str):
        """
        Update a batch of examples with attacked examples.
        """
        batch['slices'] = [rmerge(example_dict,
                                  {ATTACK: {key: [tz.valmap(lambda v: v[i], aug_batch)
                                                  for aug_batch in attacked_batches]}})
                           for i, example_dict in enumerate(batch['slices'])]
        return batch

    def process_batch(self,
                      batch: Dict[str, List],
                      keys: List[str]) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:
        slice_labels = self.apply(batch, keys)

        # Store these slice labels
        # TODO(karan): figure out how to set the alias
        batch = self.store_slice_labels(
            batch, slice_labels.tolist(), self.alias()
        )

        return batch, self.slice_batch_with_slice_labels(batch, slice_labels), slice_labels


class Subpopulation(SliceMaker):

    def __init__(
            self,
            num_slices,
            identifiers,
            slice_batch_fn=None,
    ):
        super(Subpopulation, self).__init__(
            category=SUBPOPULATION,
            num_slices=num_slices,
            identifiers=identifiers,
            slice_batch_fn=slice_batch_fn,
        )

    def apply(self,
              slice_membership: np.ndarray,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> np.ndarray:
        raise NotImplementedError

    def process_batch(self,
                      batch: Dict[str, List],
                      keys: List[str]) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:

        # Determine the size of the batch
        batch_size = len(batch[list(batch.keys())[0]])

        # Construct the matrix of slice labels: (batch_size x num_slices)
        slice_membership = np.zeros((batch_size, self.num_slices), dtype=np.int32)
        slice_membership = self.apply(slice_membership, batch, keys)

        # Store these slice labels
        # TODO(karan): figure out how to set the alias
        batch = self.store(
            batch, slice_membership.tolist(), self.alias()
        )

        return batch, self.filter_batch_by_slice_membership(batch, slice_membership), slice_membership

    @staticmethod
    def store(batch: Dict[str, List],
              slice_labels: Sequence[Sequence],
              key: str):
        """
        Update a batch of examples with slice information.
        """
        batch['slices'] = [
            rmerge(
                example_dict,
                {SUBPOPULATION: {key: slice_labels[i]}}
            )
            for i, example_dict in enumerate(batch['slices'])
        ]
        return batch

    @staticmethod
    def filter_batch_by_slice_membership(batch: Dict[str, List],
                                         slice_membership: np.ndarray) -> List[Dict[str, List]]:
        """
        Use a matrix of slice membership labels to select the subset of examples in each slice.

        Returns a list. Each element in the list corresponds to a single slice, and
        contains the subset of examples in 'batch' that lies in that slice.
        """
        return [tz.valmap(lambda v: list(compress(v, s)), batch) for s in slice_membership.T]

    @classmethod
    def union(cls, *slicers: SliceMaker):
        """
        Combine a list of slicers using a union.
        """
        # Group the slicers based on their class
        grouped_slicers = tz.groupby(lambda s: s.__class__, slicers)

        # Join the slicers corresponding to each class, and flatten
        slicers = list(tz.concat(tz.itemmap(lambda item: (item[0], item[0].join(*item[1])),
                                            grouped_slicers).values()))

        def slice_batch_fn(batch, keys):
            # Keep track of all the slice labels
            all_slice_labels = []

            # Run each slicer on the batch
            for slicer in slicers:
                # Use the batch updated by the previous slicer
                batch, _, slice_labels = slicer.process_batch(
                    batch=batch, keys=keys)
                all_slice_labels.append(slice_labels)

            # Concatenate all the slice labels
            slice_labels = np.concatenate(all_slice_labels, axis=1)

            # Take the union over the slices (columns)
            slice_labels = np.any(slice_labels, axis=1).astype(
                np.int32)[:, np.newaxis]

            return batch, cls.filter_batch_by_slice_membership(batch, slice_labels), slice_labels

        return Subpopulation(slice_batch_fn=slice_batch_fn)

    @classmethod
    def intersection(cls, *slicemakers: SliceMaker):
        """
        Combine a list of slicemakers using an intersection.
        """
        # Group the slicemakers based on their class
        grouped_slicemakers = tz.groupby(lambda s: s.__class__, slicemakers)

        # Join the slicemakers corresponding to each class, and flatten
        slicemakers = list(tz.concat(tz.itemmap(lambda item: (item[0], item[0].join(*item[1])),
                                                grouped_slicemakers).values()))

        def slice_batch_fn(batch, keys):
            # Keep track of all the slice labels
            all_slice_labels = []

            # Run each slicer on the batch
            for slicer in slicemakers:
                # Use the batch updated by the previous slicer
                batch, _, slice_labels = slicer.process_batch(
                    batch=batch, keys=keys)
                all_slice_labels.append(slice_labels)

            # Concatenate all the slice labels
            slice_labels = np.concatenate(all_slice_labels, axis=1)

            # Take the intersection over the slices (columns)
            slice_labels = np.all(slice_labels, axis=1).astype(
                np.int32)[:, np.newaxis]

            return batch, SliceMaker.slice_batch_with_slice_labels(batch, slice_labels), slice_labels

        return SliceMaker(slice_batch_fn=slice_batch_fn)


class Curator(SliceMaker):

    def __init__(
            self,
            num_slices,
            identifiers,
            apply_fn,
    ):
        super(Curator, self).__init__(
            category=CURATION,
            num_slices=num_slices,
            identifiers=identifiers,
            apply_fn=apply_fn,
        )

    def apply(self):
        pass
