import json
import pathlib
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Optional, Union, Callable

from robustnessgym.decorators import singlecolumn
from robustnessgym.core.constants import *
from robustnessgym.core.dataset import Dataset, Batch, BatchOrDataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.tools import recmerge, persistent_hash, strings_as_json


class Operation(ABC):

    def __init__(self,
                 apply_fn: Callable = None,
                 identifiers: List[Identifier] = None,
                 num_outputs: int = None,
                 *args,
                 **kwargs):

        if not identifiers:
            assert num_outputs, "Must pass in num_outputs if no identifiers are specified."

        # Set the identifiers for the outputs of the Operation
        self._identifiers = Identifier.range(
            n=num_outputs,
            _name=self.__class__.__name__,
            **kwargs
        ) if not identifiers else identifiers

        # Assign the apply_fn
        if apply_fn:
            self.apply = apply_fn

        # # Find the batch and dataset processors
        # self._batch_processors = {method.__name__ for method in
        #                           methods_with_decorator(self.__class__, batch_processing)}
        # self._dataset_processors = {method.__name__ for method in
        #                             methods_with_decorator(self.__class__, dataset_processing)}

    @property
    def identifiers(self):
        return self._identifiers

    # @property
    # @abstractmethod
    # def processors(self):
    #     raise NotImplementedError("Must specify the order in which processors are applied.")
    #
    # @property
    # def batch_processors(self):
    #     return self._batch_processors
    #
    # @property
    # def dataset_processors(self):
    #     return self._dataset_processors

    def __hash__(self):
        """
        Compute a hash value for the cached operation object.
        """
        val = 0
        for identifier in self.identifiers:
            val ^= persistent_hash(str(identifier))
        return val

    # def get_cache_hash(self,
    #                    columns: List[str],
    #                    processor: str = None):
    #     """
    #     Construct a hash that will be used to identify the application of a Operation to the columns of a dataset.
    #     """
    #
    #     # Hash the Operation
    #     val = hash(self)
    #
    #     # Combine with the hash for each column
    #     for column in columns:
    #         val ^= persistent_hash(column)
    #
    #     # Combine with the hash for the processor
    #     if processor:
    #         val ^= persistent_hash(processor)
    #
    #     return val
    #
    # def get_cache_file_name(self,
    #                         columns: List[str],
    #                         processor: str = None) -> str:
    #     """
    #     Construct a file name for caching.
    #     """
    #     return 'cache-' + str(abs(self.get_cache_hash(columns=columns, processor=processor))) + '.arrow'

    # # FIXME: temporary
    # def __call__(self,
    #              batch_or_dataset: BatchOrDataset,
    #              columns: List[str],
    #              mask: List[int] = None,
    #              *args,
    #              **kwargs) -> BatchOrDataset:
    #
    #     if isinstance(batch_or_dataset, Dataset):
    #         # Check the Dataset's InteractionTape to see if the Operation was previously applied
    #         if not mask:
    #             # This infers a mask that specifies which outputs of the Operation are not required
    #             mask = batch_or_dataset.check_tape(
    #                 path=[self.__class__.__name__],
    #                 identifiers=self.identifiers,
    #                 columns=columns
    #             )
    #
    #         # If all outputs of the Operation were previously present in the Dataset, simply return
    #         if all(mask):
    #             return batch_or_dataset
    #
    #         # Apply the CachedOperation to the dataset
    #         dataset = self.process_dataset(
    #             dataset=batch_or_dataset,
    #             columns=columns,
    #         )
    #
    #         # Update the InteractionTape with the applied CachedOperation
    #         dataset.update_tape(
    #             path=[CACHED_OPS],
    #             identifiers=self.identifiers,
    #             columns=columns,
    #         )
    #
    #         return dataset
    #
    #     elif isinstance(batch_or_dataset, Dict):
    #
    #         assert len(self.dataset_processors) == 0, \
    #             f"Cannot apply {self.__class__.__name__} to a batch, " \
    #             f"since it has dataset processors: {self.dataset_processors}. " \
    #             f"Use Dataset.from_batch(batch) before calling {self.__class__.__name__}."
    #
    #         # Apply the Operation
    #         return self.process_batch(
    #             batch=batch_or_dataset,
    #             columns=columns
    #         )
    #     else:
    #         raise NotImplementedError
    #
    # def wrap_batch_processor(self,
    #                          batch_processor: Callable) -> Callable:
    #
    #     def _wrap_batch_processor(batch: Batch,
    #                               columns: List[str],
    #                               **kwargs):
    #
    #         return batch_processor(batch=batch, columns=columns, **kwargs)
    #
    #     return _wrap_batch_processor
    #
    # def process_dataset(self,
    #                     dataset: Dataset,
    #                     columns: List[str],
    #                     batch_size: int = 32) -> Dataset:
    #     """
    #     Apply the Operation to a dataset.
    #     """
    #
    #     # Apply them in order
    #     for method in self.processors:
    #
    #         # Apply batch processors by .map(..) over the dataset
    #         if method.__name__ in self.batch_processors:
    #             dataset = dataset.map(
    #                 partial(method, columns=columns),
    #                 batched=True,
    #                 batch_size=batch_size,
    #                 cache_file_name=self.get_cache_file_name(columns=columns, processor=method)
    #             )
    #         # Apply dataset processors directly
    #         elif method.__name__ in self.dataset_processors:
    #             dataset = method(
    #                 dataset=dataset,
    #                 columns=columns,
    #             )
    #         else:
    #             raise RuntimeError(f"{method} is not a processor. "
    #                                f"Please remove {method} from the `processors` property or decorate it.")
    #
    #     return dataset
    #
    # def process_batch(self,
    #                   batch: Batch,
    #                   columns: List[str]) -> Batch:
    #     """
    #     Apply the cached operation to a batch.
    #     """
    #     assert len(set(columns) - set(batch.keys())) == 0, "Any column in 'columns' must be present in 'batch'."
    #
    #     # Run the cached operation, and encode outputs (defaults to json.dumps)
    #     encoded_outputs = [
    #         self.encode(example_output)
    #         for example_output in self.apply(batch=batch, columns=columns)
    #     ]
    #
    #     # Construct updates
    #     updates = self.construct_updates(
    #         encoded_outputs=encoded_outputs,
    #         columns=columns
    #     )
    #
    #     # Update the cache and return the updated batch
    #     return self.store(batch=batch, updates=updates)

    @classmethod
    def identify(cls, **kwargs):
        return Identifier(_name=cls.__name__, **kwargs)

    @classmethod
    def encode(cls, obj) -> str:
        return json.dumps(obj)

    @classmethod
    def decode(cls, s: str):
        return json.loads(s)

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass


class CachedOperation(Operation):
    """
    Class to create CachedOperations.
    """

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / 'robustnessgym/cachedops/'

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    def __init__(self,
                 apply_fn: Callable = None,
                 identifier: Identifier = None,
                 *args,
                 **kwargs):

        super(CachedOperation, self).__init__(
            apply_fn=apply_fn,
            identifiers=[identifier] if identifier else None,
            num_outputs=1,
            *args,
            **kwargs
        )

    @property
    def identifier(self):
        return self.identifiers[0]

    @staticmethod
    def store(batch: Batch,
              updates: List[Dict]) -> Batch:
        """
        Updates the cache of preprocessed information stored with each example in a batch.

        Args:
            batch: a batch of data
            updates: a list of dictionaries, one per example

        Returns: updated batch

        """
        if 'cache' not in batch:
            batch['cache'] = [{} for _ in range(len(batch['index']))]

        # For each example, recursively merge the example's original cache dictionary with the update dictionary
        batch['cache'] = [
            recmerge(cache_dict, update_dict)
            for cache_dict, update_dict in zip(batch['cache'], updates)
        ]

        return batch

    @classmethod
    def retrieve(cls,
                 batch: Batch,
                 columns: Union[List[str], List[List[str]]],
                 proc_fns: Union[str, Callable, List[Union[str, Callable]]] = None,
                 identifier: Union[str, Identifier] = None,
                 reapply: bool = False,
                 **kwargs,
                 ) -> Optional[Union[Batch, List[Batch]]]:
        """
        Retrieve information from the cache.

        Args:

            batch:
            columns:
            proc_fns:
            identifier:
            reapply:

        Returns:

        """

        if not reapply:
            # Nothing to return if there's no cache
            if 'cache' not in batch:
                return None

            # Infer the most relevant key to retrieve if an identifier is not specified
            if not identifier:
                for ident_key in batch['cache'][0].keys():
                    # Pick the first key that matches the cls name
                    # FIXME(karan): this is not well defined when CachedOperations are instantiated directly
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

            # Resolve the str proc_fns to callable(s)
            if isinstance(proc_fns, str):
                proc_fns = getattr(cls, proc_fns)
            elif isinstance(proc_fns, List):
                proc_fns = [proc_fn if isinstance(proc_fn, Callable) else getattr(cls, proc_fn) for proc_fn in proc_fns]

            # Process and return the retrieved information
            if isinstance(proc_fns, Callable):
                return {k: proc_fns(v) for k, v in retrieval.items()}

            return [{k: proc_fn(v) for k, v in retrieval.items()} for proc_fn in proc_fns]

        else:
            if proc_fns:
                print("Warning: proc_fns has no effect when reapply=True.")

            # Run the operation on the fly
            if isinstance(columns[0], str):
                return {strings_as_json(columns): cls(**kwargs).apply(batch=batch, columns=columns)}
            return {
                strings_as_json(cols_): cls(**kwargs).apply(batch=batch, columns=cols_)
                for cols_ in columns
            }

    def get_cache_hash(self,
                       columns: Optional[List[str]] = None):
        """
        Construct a hash that will be used to identify the application of a cached operation to the columns of a dataset.
        """

        val = hash(self)
        if columns:
            for key in columns:
                val ^= persistent_hash(key)
        return val

    def get_cache_file_name(self, columns=None):
        """
        Construct a file name for caching.
        """
        return 'cache-' + str(abs(self.get_cache_hash(columns=columns))) + '.arrow'

    def prepare_batch(self,
                      batch: Batch,
                      columns: List[str]) -> Batch:
        """
        Preparation that is applied before the CachedOperation.

        This is provided as a convenience function that can be called by prepare_dataset.

        Args:
            batch: batch of examples
            columns: list of columns

        Returns: updated batch
        """
        return batch

    def prepare_dataset(self,
                        dataset: Dataset,
                        columns: List[str],
                        batch_size: int = 32) -> Dataset:
        """
        Preparation that is applied before the CachedOperation.

        Many CachedOperations require a full pass over the dataset to precompute some variables
        before the core operation can actually be applied e.g. to create a Bag-of-Words representation,
        constructing a dataset vocabulary to keep only tokens that are frequently seen across the dataset.

        Args:
            dataset: Dataset
            columns: list of columns
            batch_size: batch size for .map(..)

        Returns: updated Dataset

        """

        # Apply preparation to the dataset
        # TODO(karan): this is similar to the try except block for slicebuilders, refactor
        try:
            return dataset.map(
                partial(self.prepare_batch, columns=columns),
                batched=True,
                batch_size=batch_size,
                cache_file_name=
                # The cache file name is a XOR of the interaction history and the current operation
                # FIXME(karan): this is repeated
                str(
                    dataset.logdir /
                    ('cache-' + str(abs(persistent_hash(str(dataset.identifier)) ^
                                        dataset.hash_interactions() ^
                                        persistent_hash(
                                            str(self.identifier) + str(strings_as_json(columns))))) + '-prep.arrow')),
            )
        except: # TypeError or PicklingError or AttributeError:
            # Batch the dataset, and process each batch
            all_batches = [
                self.prepare_batch(
                    batch=batch,
                    columns=columns,
                )
                for batch in dataset.batch(batch_size)
            ]

            return dataset.map(
                lambda examples, indices: all_batches[indices[0] // batch_size],
                batched=True,
                batch_size=batch_size,
                with_indices=True,
                load_from_cache_file=False,
                cache_file_name=
                # The cache file name is a XOR of the interaction history and the current operation
                # FIXME(karan): this is repeated
                str(
                    dataset.logdir /
                    ('cache-' + str(abs(persistent_hash(str(dataset.identifier)) ^
                                        dataset.hash_interactions() ^
                                        persistent_hash(
                                            str(self.identifier) + str(strings_as_json(columns))))) + '-prep.arrow')),
            )

    def apply(self,
              batch: Batch,
              columns: List[str],
              *args,
              **kwargs) -> List:
        """
        Implements the core functionality of the cached operation.
        """
        pass

    def process_batch(self,
                      batch: Batch,
                      columns: List[str]) -> Batch:
        """
        Apply the cached operation to a batch.
        """
        assert len(set(columns) - set(batch.keys())) == 0, "Any column in 'columns' must be present in 'batch'."

        # Run the cached operation, and encode outputs (defaults to json.dumps)
        encoded_outputs = [
            self.encode(example_output)
            for example_output in self.apply(batch=batch, columns=columns)

        ]

        # Construct updates
        updates = self.construct_updates(
            encoded_outputs=encoded_outputs,
            columns=columns
        )

        # Update the cache and return the updated batch
        return self.store(batch=batch, updates=updates)

    def process_dataset(self,
                        dataset: Dataset,
                        columns: List[str],
                        batch_size: int = 32) -> Dataset:
        """
        Apply the cached operation to a dataset.
        """

        # Prepare to apply the CachedOperation to the dataset
        dataset = self.prepare_dataset(
            dataset=dataset,
            columns=columns,
            batch_size=batch_size,
        )

        try:
            return dataset.map(
                partial(self.process_batch, columns=columns),
                batched=True,
                batch_size=batch_size,
                cache_file_name=
                # The cache file name is a XOR of the interaction history and the current operation
                str(dataset.logdir /
                    ('cache-' + str(abs(persistent_hash(str(dataset.identifier)) ^
                                        dataset.hash_interactions() ^
                                        persistent_hash(
                                            str(self.identifier) + str(strings_as_json(columns))))) + '.arrow')),
                # self.get_cache_file_name(columns=columns),
            )
        except:
            # Batch the dataset, and process each batch
            all_batches = [
                self.process_batch(
                    batch=batch,
                    columns=columns,
                )
                for batch in dataset.batch(batch_size)
            ]

            return dataset.map(
                lambda examples, indices: all_batches[indices[0] // batch_size],
                batched=True,
                batch_size=batch_size,
                with_indices=True,
                load_from_cache_file=False,
                cache_file_name=
                # The cache file name is a XOR of the interaction history and the current operation
                str(dataset.logdir /
                    ('cache-' + str(abs(persistent_hash(str(dataset.identifier)) ^
                                        dataset.hash_interactions() ^
                                        persistent_hash(
                                            str(self.identifier) + str(strings_as_json(columns))))) + '.arrow')),
            )

    def construct_updates(self,
                          encoded_outputs: List[str],
                          columns: List[str]):
        return [{
            str(self.identifier): {strings_as_json(columns): val}
        } for val in encoded_outputs]

    @classmethod
    def available(cls, batch: Batch):
        # Check if the cached operation is available to retrieve in the batch
        if 'cache' not in batch:
            return False
        return any([key.startswith(cls.__name__) for key in batch['cache'][0].keys()])

    def __call__(self,
                 batch_or_dataset: BatchOrDataset,
                 columns: List[str],
                 batch_size: int = 32) -> BatchOrDataset:

        if isinstance(batch_or_dataset, Dataset):

            # Check the InteractionTape to see if the CachedOperation was applied
            if batch_or_dataset.check_tape(
                    path=[CACHED_OPS],
                    identifiers=self.identifier,
                    columns=columns,
            ):
                return batch_or_dataset

            # Apply the CachedOperation to the dataset
            dataset = self.process_dataset(
                dataset=batch_or_dataset,
                columns=columns,
                batch_size=batch_size,
            )

            # Update the InteractionTape with the applied CachedOperation
            dataset.update_tape(
                path=[CACHED_OPS],
                identifiers=self.identifier,
                columns=columns,
            )

            return dataset

        elif isinstance(batch_or_dataset, Dict):

            # Apply the CachedOperation
            return self.process_batch(
                batch=batch_or_dataset,
                columns=columns
            )
        else:
            raise NotImplementedError


class SingleColumnCachedOperation(CachedOperation):

    def __call__(self,
                 batch_or_dataset: BatchOrDataset,
                 columns: List[str],
                 batch_size: int = 32) -> BatchOrDataset:
        """
        Apply independently to each column.

        Args:
            batch_or_dataset:
            columns:

        Returns:

        """
        # Iterate over the columns and apply
        for column in columns:
            batch_or_dataset = super(SingleColumnCachedOperation, self).__call__(
                batch_or_dataset=batch_or_dataset,
                columns=[column],
                batch_size=batch_size,
            )

        return batch_or_dataset

    @singlecolumn
    def apply(self,
              batch: Batch,
              columns: List[str],
              *args,
              **kwargs) -> List:
        return self.single_column_apply(batch[columns[0]])

    def single_column_apply(self,
                            column_batch: List,
                            **kwargs) -> List:
        raise NotImplementedError("Must implement single_column_apply.")


class ScoreOperation(CachedOperation):

    def apply(self,
              batch: Batch,
              columns: List[str],
              *args,
              **kwargs) -> List[Union[int, float]]:
        return super().apply(batch, columns, *args, **kwargs)


def stow(dataset: Dataset,
         cached_ops: Dict[CachedOperation, List[List[str]]],
         batch_size: int = 32,
         load_from_cache_file: bool = True):
    """
    Apply a list of cached operations in sequence.
    """

    # Check the InteractionTape to remove CachedOperations that have already been stowed
    for cached_op, list_of_columns in list(cached_ops.items()):
        indices_to_remove = []
        for i, columns in enumerate(list(list_of_columns)):
            if dataset.check_tape(
                    path=[CACHED_OPS],
                    identifiers=cached_op.identifier,
                    columns=columns,
            ):
                # Remove the columns at index i
                indices_to_remove.append(i)

        # Remove the columns that are already cached
        for index in sorted(indices_to_remove, reverse=True):
            columns = cached_ops[cached_op].pop(index)
            print(f"skipped: {cached_op.identifier} -> {columns}", flush=True)

        # Check if list_of_columns is now empty
        if not cached_ops[cached_op]:
            # Remove the op entirely
            cached_ops.pop(cached_op)

    for cached_op, list_of_columns in cached_ops.items():
        for columns in list_of_columns:
            dataset = cached_op(dataset, columns=columns, batch_size=batch_size)

    # def _map_fn(batch: Batch):
    #     """
    #     Consolidate the application of the CachedOperations passed to stow into a single mappable function.
    #     """
    #     for cached_op, list_of_columns in cached_ops.items():
    #         for columns in list_of_columns:
    #             batch = cached_op(batch, columns=columns)
    #
    #     return batch
    #
    # # Compute the hash value
    # val = 0
    # for cached_op, list_of_columns in cached_ops.items():
    #     for columns in list_of_columns:
    #         val ^= cached_op.get_cache_hash(columns=columns)
    #
    # # Combine with the hash for the dataset on which the cached ops are applied
    # val ^= persistent_hash(
    #     # TODO(karan): move this to Dataset
    #     "-".join(
    #         "-".join(str(k) + "-" + str(v) for k, v in f.items()) for f in dataset._data_files
    #     )
    # )
    #
    # # Map the cached operations over the dataset
    # try:
    #     dataset = dataset.map(
    #         _map_fn,
    #         batched=True,
    #         batch_size=32,
    #         cache_file_name='cache-' + str(abs(val)) + '.arrow',
    #         load_from_cache_file=load_from_cache_file
    #     )
    # except TypeError:
    #     # Batch the dataset, and process each batch
    #     all_batches = [_map_fn(batch=batch) for batch in dataset.batch(batch_size)]
    #
    #     # Update the dataset efficiently by reusing all_batches
    #     dataset = dataset.map(
    #         lambda examples, indices: all_batches[indices[0] // batch_size],
    #         batched=True,
    #         batch_size=batch_size,
    #         with_indices=True,
    #     )

    # Update the Dataset history
    for cached_op, list_of_columns in cached_ops.items():
        for columns in list_of_columns:
            dataset.update_tape(
                path=[CACHED_OPS],
                identifiers=cached_op.identifier,
                columns=columns
            )

    return dataset
