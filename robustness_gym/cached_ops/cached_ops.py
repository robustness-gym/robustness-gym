import json
import pathlib
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Optional, Union, Callable

from robustness_gym.constants import *
from robustness_gym.dataset import Dataset
from robustness_gym.identifier import Identifier
from robustness_gym.tools import recmerge, persistent_hash, strings_as_json


class Operation(ABC):

    def __init__(self,
                 apply_fn: Callable = None,
                 *args,
                 **kwargs):
        # Set the identifier for the operation
        self._identifier = Identifier(name=self.__class__.__name__, **kwargs)

        # Assign the apply_fn
        if apply_fn:
            self.apply = apply_fn

    def __hash__(self):
        """
        Compute a hash value for the cached operation object.
        """
        return persistent_hash(str(self.identifier))

    @property
    def identifier(self):
        return self._identifier

    @classmethod
    def identify(cls, **kwargs):
        return Identifier(name=cls.__name__, **kwargs)

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
    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / 'robustnessgym/cachedops/'

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    def __init__(self,
                 apply_fn: Callable = None,
                 *args,
                 **kwargs,
                 ):

        super(CachedOperation, self).__init__(
            apply_fn=apply_fn,
            *args,
            **kwargs,
        )

    @staticmethod
    def store(batch: Dict[str, List],
              updates: List[Dict]) -> Dict[str, List]:
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
                 batch: Dict[str, List],
                 keys: Union[List[str], List[List[str]]],
                 proc_fns: Union[str, Callable, List[Union[str, Callable]]] = None,
                 identifier: Union[str, Identifier] = None,
                 reapply: bool = False,
                 **kwargs,
                 ) -> Optional[Union[Dict[str, List], List[Dict[str, List]]]]:
        """
        Retrieve information from the cache.

        Args:

            batch:
            keys:
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
                    if ident_key.startswith(cls.__name__):
                        identifier = ident_key
                        break
            try:
                if isinstance(keys[0], str):
                    retrieval = {
                        strings_as_json(keys): [
                            cls.decode(cache[str(identifier)][strings_as_json(keys)]) for cache in batch['cache']
                        ]
                    }
                else:
                    retrieval = {
                        strings_as_json(keys_): [
                            cls.decode(cache[str(identifier)][strings_as_json(keys_)]) for cache in batch['cache']
                        ]
                        for keys_ in keys
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
            if isinstance(keys[0], str):
                return {strings_as_json(keys): cls(**kwargs).apply(*[batch[key] for key in keys])}
            return {
                strings_as_json(keys_): cls(**kwargs).apply(*[batch[key] for key in keys_])
                for keys_ in keys
            }

    def get_cache_hash(self,
                       keys: Optional[List[str]] = None):
        """
        Construct a hash that will be used to identify the application of a cached operation to the keys of a dataset.
        """

        val = hash(self)
        if keys:
            for key in keys:
                val ^= persistent_hash(key)
        return val

    def get_cache_file_name(self, keys=None):
        """
        Construct a file name for caching.
        """
        return 'cache-' + str(abs(self.get_cache_hash(keys=keys))) + '.arrow'

    def process_dataset(self,
                        dataset: Dataset,
                        keys: List[str],
                        batch_size: int = 32) -> Dataset:
        """
        Apply the cached operation to a dataset.
        """

        # Apply the cached operation
        return dataset.map(
            partial(self.process_batch, keys=keys),
            batched=True,
            batch_size=batch_size,
            cache_file_name=self.get_cache_file_name(keys=keys)
        )

    def process_batch(self,
                      batch: Dict[str, List],
                      keys: List[str]) -> Dict[str, List]:
        """
        Apply the cached operation to a batch.
        """
        assert len(set(keys) - set(batch.keys())) == 0, "Any key in 'keys' must be present in 'batch'."

        # Run the cached operation, and encode outputs (defaults to json.dumps)
        encoded_outputs = [
            self.encode(example_output)
            for example_output in self.apply(*[batch[key] for key in keys])
        ]

        # Construct updates
        updates = [{
            # TODO(karan): Update this to handle identifiers: figure out how to access inside slicemakers
            str(self.identifier):
            # self.__class__.__name__:
                {
                    strings_as_json(keys): val
                }
        }
            for val in encoded_outputs]

        # Update the cache and return the updated batch
        return self.store(batch=batch, updates=updates)

    def apply(self, *args, **kwargs) -> List:
        """
        Implements the core functionality of the cached operation.
        """
        pass

    @classmethod
    def available(cls, batch: Dict[str, List]):
        # Check if the cached operation is available to retrieve in the batch
        if 'cache' not in batch:
            return False
        return any([key.startswith(cls.__name__) for key in batch['cache'][0].keys()])

    def __call__(self,
                 batch_or_dataset: Union[Dict[str, List], Dataset],
                 keys: List[str]):

        if isinstance(batch_or_dataset, Dataset):
            # Check the InteractionTape to see if the CachedOperation was applied
            if not batch_or_dataset.check_tape(
                    path=[CACHED_OPS],
                    identifier=self.identifier,
                    keys=keys,
            ):
                return batch_or_dataset

            # Apply the CachedOperation to the dataset
            dataset = self.process_dataset(
                dataset=batch_or_dataset,
                keys=keys,
            )

            # Update the InteractionTape with the applied CachedOperation
            dataset.update_tape(
                path=[CACHED_OPS],
                identifier=self.identifier,
                keys=keys,
            )

            return dataset

        elif isinstance(batch_or_dataset, Dict):
            # Apply the CachedOperation to a batch
            return self.process_batch(batch=batch_or_dataset,
                                      keys=keys)
        else:
            raise NotImplementedError


def stow(dataset: Dataset,
         cached_ops: Dict[CachedOperation, List[List[str]]],
         load_from_cache_file: bool = True):
    """
    Apply a list of cached operations in sequence.
    """

    # Check the InteractionTape to remove CachedOperations that have already been stowed
    for cached_op, list_of_keys in list(cached_ops.items()):
        indices_to_remove = []
        for i, keys in enumerate(list(list_of_keys)):
            if not dataset.check_tape(
                    path=[CACHED_OPS],
                    identifier=cached_op.identifier,
                    keys=keys
            ):
                # Remove the keys at index i
                indices_to_remove.append(i)

        # Remove the keys that are already cached
        for index in sorted(indices_to_remove, reverse=True):
            keys = cached_ops[cached_op].pop(index)
            print(f"skipped: {cached_op.identifier} -> {keys}", flush=True)

        # Check if list_of_keys is now empty
        if not cached_ops[cached_op]:
            # Remove the op entirely
            cached_ops.pop(cached_op)

    def _map_fn(batch: Dict[str, List]):
        """
        Consolidate the application of the CachedOperations passed to stow into a single mappable function.
        """
        for cached_op, list_of_keys in cached_ops.items():
            for keys in list_of_keys:
                batch = cached_op.process_batch(batch, keys=keys)

        return batch

    # Compute the hash value
    val = 0
    for cached_op, list_of_keys in cached_ops.items():
        for keys in list_of_keys:
            val ^= cached_op.get_cache_hash(keys=keys)

    # Combine with the hash for the dataset on which the cached ops are applied
    val ^= persistent_hash(
        # TODO(karan): move this to Dataset
        "-".join(
            "-".join(str(k) + "-" + str(v) for k, v in f.items()) for f in dataset._data_files
        )
    )

    # Map the cached operations over the dataset
    dataset = dataset.map(
        _map_fn,
        batched=True,
        batch_size=32,
        cache_file_name='cache-' + str(abs(val)) + '.arrow',
        load_from_cache_file=load_from_cache_file
    )

    # Update the Dataset history
    for cached_op, list_of_keys in cached_ops.items():
        for keys in list_of_keys:
            dataset.update_tape(
                path=[CACHED_OPS],
                identifier=cached_op.identifier,
                keys=keys
            )

    return dataset
