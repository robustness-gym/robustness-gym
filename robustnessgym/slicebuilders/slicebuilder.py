from __future__ import annotations
import tqdm


def nop(it, *a, **k):
    return it


tqdm.tqdm = nop
import contextlib
import os
import pathlib
import time
from functools import partial
from itertools import compress
from multiprocess.pool import Pool
from pickle import PicklingError
from typing import *

import cytoolz as tz
import numpy as np
# from tqdm import tqdm

from robustnessgym.cached_ops.cached_ops import CachedOperation
from robustnessgym.constants import *
from robustnessgym.dataset import Dataset, Batch
from robustnessgym.identifier import Identifier
from robustnessgym.slice import Slice
from robustnessgym.storage import StorageMixin
from robustnessgym.tools import recmerge, strings_as_json, persistent_hash


class SliceBuilder(StorageMixin):
    """
    Base class for builders that output slices.
    """

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / 'robustnessgym/slicemakers/'

    # Create the log directory
    logdir.mkdir(parents=True, exist_ok=True)

    CATEGORIES = [
        GENERIC,
        SUBPOPULATION,
        ATTACK,
        AUGMENTATION,
        CURATION,
    ]

    def __init__(self,
                 category: str,
                 identifiers: List[Identifier],
                 apply_fn: Callable = None,
                 *args,
                 **kwargs):

        super(SliceBuilder, self).__init__(*args, **kwargs)

        # The SliceMaker belongs to a category
        assert category in self.CATEGORIES, f"argument category must be one of {self.CATEGORIES}"
        self.category = category

        # Each identifier corresponds to a single output Slice generated by this SliceBuilder
        self.identifiers = identifiers

        # Keep track of the CachedOperation dependencies
        self.prerequisites = set()
        for base in self.__class__.__bases__:
            for cls in base.__mro__:
                if str(CachedOperation.__name__) in str(cls):
                    self.prerequisites.add(base)

        if apply_fn:
            # Assign to the method
            self.apply = apply_fn

    def __call__(self,
                 batch_or_dataset: Union[Dict[str, List], Dataset],
                 columns: List[str],
                 mask: List[int] = None,
                 store_compressed: bool = None,
                 store: bool = None,
                 num_proc: int = None,
                 *args,
                 **kwargs):

        # Check that prerequisites are satisfied
        self.prerequisites_handler(batch_or_dataset)

        if isinstance(batch_or_dataset, Dataset):

            # Slice a dataset
            dataset, slices, slice_membership = self.process_dataset(
                dataset=batch_or_dataset,
                columns=columns,
                # Automatically infer the mask from the Dataset if it's not specified
                mask=batch_or_dataset.check_tape(
                    path=[SLICEMAKERS, self.category],
                    identifiers=self.identifiers,
                    columns=columns
                ) if not mask else mask,
                store_compressed=True if store_compressed is None else store_compressed,
                store=True if store is None else store,
                num_proc=num_proc,
                *args,
                **kwargs
            )

            # Update the Dataset's history
            # TODO(karan): use mask to figure out what is actually applied
            dataset.update_tape(
                path=[SLICEMAKERS, self.category],
                identifiers=self.identifiers,
                columns=columns,
            )

            return dataset, slices, slice_membership

        elif isinstance(batch_or_dataset, Dict):
            if store_compressed is True:
                print("Compressed storage cannot be used on a batch. Please use Dataset.from_batch(batch) before "
                      "applying the SliceBuilder.")
            # Slice a batch
            return self.process_batch(
                batch=batch_or_dataset,
                columns=columns,
                mask=mask,
                # Don't allow compressed storage for __call__ on a batch
                store_compressed=False,
                # Don't store by default
                store=False if store is None else store,
                *args,
                **kwargs
            )
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{self.category}[{self.__class__.__name__}(num_slices={self.num_slices})]"

    @property
    def num_slices(self):
        return len(self.identifiers)

    def __getitem__(self, item: int):
        return self.identifiers[item]

    def __iter__(self):
        yield from self.identifiers

    def prerequisites_handler(self,
                              batch_or_dataset: Union[Dict[str, List], Dataset]):
        if isinstance(batch_or_dataset, Dataset):
            batch = batch_or_dataset[:2]
        else:
            batch = batch_or_dataset

        # Check if pre-requisites are satisfied
        # TODO(karan): move to a method
        if 'cache' not in batch:
            pending = self.prerequisites
        else:
            pending = {prerequisite for prerequisite in self.prerequisites
                       if not prerequisite.available(batch)}

        # TODO(karan): Automatically run the pending pre-requisites
        if pending:
            raise RuntimeError(f"Cannot run SliceBuilder, prerequisites {pending} not satisfied.")

    @staticmethod
    def store(batch: Dict[str, List],
              updates: List[Dict]) -> Dict[str, List]:
        """
        Update a batch of examples with slice information.
        """
        if 'slices' not in batch:
            batch['slices'] = [{} for _ in range(len(batch['index']))]

        # For each example, recursively merge the example's original cache dictionary with the update dictionary
        batch['slices'] = [
            recmerge(example_dict, update_dict, merge_sequences=True)
            for example_dict, update_dict in zip(batch['slices'], updates)
        ]

        return batch

    def prepare_batch(self,
                      batch: Batch,
                      columns: List[str],
                      mask: List[int] = None,
                      store_compressed: bool = True,
                      store: bool = True,
                      *args,
                      **kwargs) -> Batch:
        return batch

    def prepare_dataset(self,
                        dataset: Dataset,
                        columns: List[str],
                        batch_size: int = 32,
                        mask: List[int] = None,
                        store_compressed: bool = True,
                        store: bool = True,
                        *args,
                        **kwargs) -> Dataset:

        # Compute the hash for this operation
        # FIXME(karan): this is repeated inside process_dataset
        val = persistent_hash(str(dataset.identifier)) ^ dataset.hash_interactions()
        for i, identifier in enumerate(self.identifiers):
            if not mask[i]:
                val ^= persistent_hash(str(identifier) + str(strings_as_json(columns)))

        try:
            return dataset.map(
                partial(self.prepare_batch,
                        columns=columns,
                        mask=mask,
                        store_compressed=store_compressed,
                        store=store,
                        *args,
                        **kwargs),
                batched=True,
                batch_size=batch_size,
                load_from_cache_file=False,
                cache_file_name=str(dataset.logdir / ('cache-' + str(abs(val)) + '-prep.arrow')),
            )
        except:  # TypeError or PicklingError or AttributeError:
            # Batch the dataset, and process each batch
            all_batches = [self.prepare_batch(
                batch=batch,
                columns=columns,
                mask=mask,
                store_compressed=store_compressed,
                store=store,
                *args,
                **kwargs
            )
                for batch in dataset.batch(batch_size)
            ]

            # Update the dataset efficiently by reusing all_batches
            return dataset.map(
                lambda examples, indices: all_batches[indices[0] // batch_size],
                batched=True,
                batch_size=batch_size,
                with_indices=True,
                load_from_cache_file=False,
                cache_file_name=str(dataset.logdir / ('cache-' + str(abs(val)) + '-prep.arrow')),
            )

    def process_dataset(self,
                        dataset: Dataset,
                        columns: List[str],
                        batch_size: int = 32,
                        mask: List[int] = None,
                        store_compressed: bool = True,
                        store: bool = True,
                        num_proc: int = None,
                        *args,
                        **kwargs) -> Tuple[Dataset, List[Slice], np.ndarray]:
        """
        Apply a SliceBuilder to a dataset.

        Args:
            dataset: Dataset
            columns: list of columns
            batch_size: integer batch size
            mask: boolean or integer mask array, mask[i] = True means that the ith slice will be masked out
            store_compressed: whether to store in a compressed format
            store: whether to store the results along with the example in Dataset
            num_proc: num processes for multiprocessing
            *args: optional additional arguments
            **kwargs: optional additional keyword arguments

        Returns: tuple of (Dataset, list of Slices, matrix of (example, slice) membership)

        """
        # Prepare the dataset
        dataset = self.prepare_dataset(
            dataset=dataset,
            columns=columns,
            batch_size=batch_size,
            mask=mask,
            store_compressed=store_compressed,
            store=store,
            *args,
            **kwargs
        )

        all_sliced_batches = []
        all_slice_memberships = []

        def _map_fn(batch):
            batch, sliced_batches, slice_membership = self.process_batch(
                batch=batch,
                columns=columns,
                mask=mask,
                store_compressed=store_compressed,
                store=store,
                *args,
                **kwargs
            )
            all_sliced_batches.append(sliced_batches)
            all_slice_memberships.append(slice_membership)
            return batch

        # Map the SliceBuilder over the dataset
        val = persistent_hash(str(dataset.identifier)) ^ dataset.hash_interactions()
        for i, identifier in enumerate(self.identifiers):
            if not mask[i]:
                val ^= persistent_hash(str(identifier) + str(strings_as_json(columns)))

        dataset = dataset.map(
            _map_fn,
            batched=True,
            batch_size=batch_size,
            # FIXME(karan): enable this by adding logic for generating all_sliced_batches and all_slice_memberships
            #  when loading from cache file
            load_from_cache_file=False,
            cache_file_name=
            # The cache file name is a XOR of the interaction history and the current operation
            str(dataset.logdir / ('cache-' + str(abs(val)) + '.arrow')),
        )

        # Create a single slice label matrix
        slice_membership = np.concatenate(all_slice_memberships, axis=0)

        print(' ')

        slice_cache_hashes = []
        for identifier in self.identifiers:
            slice_cache_hashes.append(val ^ persistent_hash(str(identifier)))

        if not num_proc or num_proc == 1:
            # Construct slices
            slices = []
            for i, slice_batches in enumerate(zip(*all_sliced_batches)):
                slices.append(create_slice((dataset, slice_membership, slice_batches, i, batch_size,
                                            slice_cache_hashes[i])))
        else:
            # Parallelized slice construction
            with Pool(num_proc) as pool:
                slices = pool.map(
                    create_slice,
                    [(dataset, slice_membership, slice_batches, i, batch_size, slice_cache_hashes[i])
                     for i, slice_batches in enumerate(zip(*all_sliced_batches))]
                )

        # TODO(karan): make this more systematic
        for i, sl in enumerate(slices):
            # # Set the Slice features
            # sl.info.features = dataset.features

            # Set the Slice category using the SliceBuilder's category
            sl.category = self.category

            # Create the lineage
            sl.lineage = [
                (str(Dataset.__name__), dataset.identifier),
                (str(self.category.capitalize()), self.identifiers[i])
            ]
            if isinstance(dataset, Slice):
                # Prepend the Slice's lineage instead, if the dataset was a slice
                sl.lineage = dataset.lineage + (str(self.category.capitalize()), self.identifiers[i])

        return dataset, slices, slice_membership

    def process_batch(self,
                      batch: Dict[str, List],
                      columns: List[str],
                      mask: List[int] = None,
                      store_compressed: bool = True,
                      store: bool = True,
                      *args,
                      **kwargs) \
            -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:
        """
        Apply a SliceBuilder to a batch of data.

        Args:
            batch: a batch of data
            columns: list of columns
            mask: boolean or integer mask array, mask[i] = True means that the ith slice will be masked out
            store_compressed: whether to store in a compressed format
            store: whether to store the results along with the example in Dataset
            *args: optional additional arguments
            **kwargs: optional additional keyword arguments

        Returns: tuple of (batch, list of slices (as batches), matrix of (example, slice) membership))

        """
        return batch, [batch], None

    def postprocess_dataset(self,
                            dataset: Dataset,
                            columns: List[str],
                            batch_size: int = 32) -> Dataset:
        pass

    def apply(self, *args, **kwargs):
        raise NotImplementedError("Must implement apply.")

    @classmethod
    def join(cls, *slicemakers: SliceBuilder) -> Sequence[SliceBuilder]:
        """
        Join many slicemakers. By default, just returns the slicemakers.
        """
        return slicemakers

    def masked(self, mask: List[int]):
        pass

    def unmasked(self):
        pass

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
    def retrieve(cls,
                 batch: Batch,
                 columns: Union[List[str], List[List[str]]],
                 proc_fns: Union[str, Callable, List[Union[str, Callable]]] = None,
                 identifier: Union[str, Identifier] = None,
                 reapply: bool = False,
                 **kwargs,
                 ) -> Optional[Union[Batch, List[Batch]]]:
        if not reapply:
            if 'slices' not in batch:
                return None

            # Infer the most relevant key to retrieve if an identifier is not specified
            if not identifier:
                for ident_key in batch['slices'][0].keys():
                    # Pick the first key that matches the cls name
                    if ident_key.startswith(cls.__name__):
                        identifier = ident_key
                        break

            try:
                if isinstance(columns[0], str):
                    retrieval = {
                        strings_as_json(columns): [
                            cls.decode(cache[str(identifier)][strings_as_json(columns)]) for cache in batch['cache']
                        ]
                    }
                else:
                    retrieval = {
                        strings_as_json(cols_): [
                            cls.decode(cache[str(identifier)][strings_as_json(cols_)]) for cache in batch['cache']
                        ]
                        for cols_ in columns
                    }
            except KeyError:
                raise ValueError('Could not retrieve information for all keys.')

            # Check if the retrieved information needs to be processed
            if not proc_fns:
                return retrieval
            pass
        else:
            pass


class SliceBuilderCollection(SliceBuilder):

    def __init__(self,
                 slicebuilders: List[SliceBuilder],
                 *args,
                 **kwargs):
        super(SliceBuilderCollection, self).__init__(
            category=GENERIC,
            identifiers=list(tz.concat([slicebuilder.identifiers for slicebuilder in slicebuilders])),
            *args,
            **kwargs
        )

        # TODO(karan): some slicebuilders aren't compatible with each other (e.g. single column vs. multi column):
        #  add some smarter logic here to handle this

        # Store the subpopulations
        self.slicebuilders = slicebuilders

    def __repr__(self):
        # TODO(karan): format this nicely
        return f"{self.__class__.__name__}({[str(slicebuilder) for slicebuilder in self.slicebuilders]})]"

    def __call__(self,
                 batch_or_dataset: Union[Dict[str, List], Dataset],
                 columns: List[str],
                 mask: List[int] = None,
                 store_compressed: bool = None,
                 store: bool = None,
                 *args,
                 **kwargs):

        if mask:
            raise NotImplementedError("Mask not supported for SliceBuilderCollection yet.")

        slices = []
        slice_membership = []
        # Apply each slicebuilder in sequence
        for i, slicebuilder in tqdm.tqdm(enumerate(self.slicebuilders)):
            # Apply the slicebuilder
            batch_or_dataset, slices_i, slice_membership_i = slicebuilder(batch_or_dataset=batch_or_dataset,
                                                                          columns=columns,
                                                                          mask=mask,
                                                                          store_compressed=store_compressed,
                                                                          store=store,
                                                                          *args,
                                                                          **kwargs)

            # Add in the slices and slice membership
            slices.extend(slices_i)
            slice_membership.append(slice_membership_i)

        slice_membership = np.concatenate(slice_membership, axis=1)

        return batch_or_dataset, slices, slice_membership


def create_slice(args):
    # Unpack args
    dataset, slice_membership, slice_batches, i, batch_size, slice_cache_hash = args

    # Create a new empty slice
    sl = Slice.from_dict({})

    # Create a Slice "copy" of the Dataset
    sl.__dict__.update(dataset.__dict__)

    # Filter
    sl = sl.filter(
        lambda example, idx: bool(slice_membership[idx, i]),
        with_indices=True,
        input_columns=['index'],
        batch_size=batch_size,
        cache_file_name=str(dataset.logdir / ('cache-' + str(abs(slice_cache_hash)) + '-filter.arrow'))
    )

    slice_batch = tz.merge_with(tz.compose(list, tz.concat), slice_batches)

    # FIXME(karan): interaction tape history is wrong here, esp with augmenation/attacks

    # Map
    if len(sl):
        sl = sl.map(
            lambda batch, indices: tz.valmap(lambda v: v[indices[0]: indices[0] + batch_size], slice_batch),
            batched=True,
            batch_size=batch_size,
            with_indices=True,
            remove_columns=sl.column_names,
            cache_file_name=str(dataset.logdir / ('cache-' + str(abs(slice_cache_hash)) + '.arrow')),
        )

    return sl
