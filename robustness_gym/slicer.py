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
from robustness_gym.dataset import Dataset
from robustness_gym.slice import Slice
import robustness_gym
import dill as pickle


class Slicer:

    def __init__(self,
                 slice_batch_fn=None,
                 *args,
                 **kwargs):
        super(Slicer, self).__init__(*args, **kwargs)

        self.type = None
        self.tasks = None
        self.metadata = {}
        self.num_slices = None

        if slice_batch_fn:
            # Assign to the method
            self.slice_batch = slice_batch_fn

    def __call__(self, *args, **kwargs):

        assert 'keys' in kwargs, "keys must be a keyword argument."

        if len(args) == 1:
            if isinstance(args[0], Mapping):
                # Assume it's a batch
                return self.slice_batch(batch=args[0], keys=kwargs['keys'])

            elif isinstance(args[0], nlp.Dataset):
                # Slice a dataset
                return self.slice_dataset(dataset=args[0], keys=kwargs['keys'])

            else:
                raise NotImplementedError
        else:
            if 'batch' in kwargs:
                # Slice a batch
                return self.slice_batch(batch=kwargs['batch'], keys=kwargs['keys'])
            elif 'dataset' in kwargs:
                # Slice a dataset
                return self.slice_dataset(dataset=kwargs['dataset'], keys=kwargs['keys'])
            else:
                raise NotImplementedError

    def alias(self) -> str:
        return self.__class__.__name__

    def slice_dataset(self,
                      dataset: Dataset,
                      keys: List[str],
                      batch_size: int = 32) -> Tuple[Dataset, List[Slice], np.ndarray]:
        """
        Slice a dataset.
        """

        # Batch the dataset, and slice each batch
        all_batches, all_sliced_batches, all_slice_labels = zip(
            *[self.slice_batch(tz.merge_with(tz.identity, *batch), keys)
              for batch in tz.partition_all(batch_size, dataset)])

        # TODO(karan): want to do this instead but .map() in Huggingface nlp must return either a None type or dict
        # all_batches, all_sliced_batches, all_slice_labels = \
        #     zip(*dataset.map(lambda examples: self.slice_batch(batch=examples, keys=keys),
        #                      batched=True, batch_size=batch_size))

        # Update the dataset efficiently by reusing all_batches
        # TODO(karan): have to run this separately since .map() only updates the dataset if a dict is returned
        dataset = dataset.map(lambda examples, indices: all_batches[indices[0] // batch_size],
                              batched=True, batch_size=batch_size, with_indices=True)

        # Create the dataset slices
        slices = [Slice.from_batches(slice_batches) for slice_batches in zip(*all_sliced_batches)]

        # Create a single slice label matrix
        slice_labels = np.concatenate(all_slice_labels, axis=0)

        return dataset, slices, slice_labels

    @staticmethod
    def slice_batch_with_slice_labels(batch: Dict[str, List],
                                      slice_labels: np.ndarray) -> List[Dict[str, List]]:
        """
        Use a matrix of slice labels to select the subset of examples in each slice.

        Returns a list. Each element in the list corresponds to a single slice, and
        contains the subset of examples in 'batch' that lies in that slice.
        """
        return [tz.valmap(lambda v: list(compress(v, s)), batch) for s in slice_labels.T]

    def slice_batch(self,
                    batch: Dict[str, List],
                    keys: List[str]) -> Tuple[List[Dict[str, List]], Optional[np.ndarray]]:
        pass

    @classmethod
    def join(cls, *slicers: Slicer) -> Sequence[Slicer]:
        """
        Join many slicers. By default, just returns the slicers.
        """
        return slicers

    def save(self, path: str) -> None:
        """
        Save a Slicer.
        """
        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def load(cls, path: str) -> Slicer:
        """
        Load a Slicer from a path.
        """
        return pickle.load(open(path, 'rb'))


class AugmentationMixin:

    def __init__(self):
        super(AugmentationMixin, self).__init__()

        self.type = 'augmentation'

    @staticmethod
    def store_augmentations(batch: Dict[str, List],
                            augmented_batches: List[Dict[str, List]],
                            key: str):
        """
        Update a batch of examples with augmented examples.
        """
        batch['slices'] = [rmerge(example_dict,
                                  {'augmented': {key: [tz.valmap(lambda v: v[i], aug_batch)
                                                       for aug_batch in augmented_batches]}})
                           for i, example_dict in enumerate(batch['slices'])]
        return batch


class AdversarialAttackMixin:

    def __init__(self):
        super(AdversarialAttackMixin, self).__init__()


class FilterMixin:

    def __init__(self):
        super(FilterMixin, self).__init__()

        self.type = 'filter'

    @staticmethod
    def store_slice_labels(batch: Dict[str, List],
                           slice_labels: Sequence[Sequence],
                           key: str):
        """
        Update a batch of examples with slice information.
        """
        batch['slices'] = [rmerge(example_dict, {'filtered': {key: slice_labels[i]}})
                           for i, example_dict in enumerate(batch['slices'])]
        return batch

    @classmethod
    def union(cls, *slicers: Slicer):
        """
        Combine a list of slicers using a union.
        """
        # Group the slicers based on their class
        grouped_slicers = tz.groupby(lambda s: s.__class__, slicers)

        # Join the slicers corresponding to each class, and flatten
        slicers = list(tz.concat(tz.itemmap(lambda item: (item[0], item[0].join(*item[1])),
                                            grouped_slicers).values()))

        def slice_batch_fn(batch, keys):
            print(batch['slices'][0]['filtered'])
            # Keep track of all the slice labels
            all_slice_labels = []

            # Run each slicer on the batch
            for slicer in slicers:
                # Use the batch updated by the previous slicer
                batch, _, slice_labels = slicer.slice_batch(batch=batch, keys=keys)
                all_slice_labels.append(slice_labels)

            # Concatenate all the slice labels
            slice_labels = np.concatenate(all_slice_labels, axis=1)

            # Take the union over the slices (columns)
            slice_labels = np.any(slice_labels, axis=1).astype(np.int32)[:, np.newaxis]

            return batch, Slicer.slice_batch_with_slice_labels(batch, slice_labels), slice_labels

        return Slicer(slice_batch_fn=slice_batch_fn)

    @classmethod
    def intersection(cls, *slicers: Slicer):
        """
        Combine a list of slicers using an intersection.
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
                batch, _, slice_labels = slicer.slice_batch(batch=batch, keys=keys)
                all_slice_labels.append(slice_labels)

            # Concatenate all the slice labels
            slice_labels = np.concatenate(all_slice_labels, axis=1)

            # Take the intersection over the slices (columns)
            slice_labels = np.all(slice_labels, axis=1).astype(np.int32)[:, np.newaxis]

            return batch, Slicer.slice_batch_with_slice_labels(batch, slice_labels), slice_labels

        return Slicer(slice_batch_fn=slice_batch_fn)
