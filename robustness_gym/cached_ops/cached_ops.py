import json
from functools import partial
from typing import Dict, List, Optional, Union

from robustness_gym.constants import *
from robustness_gym.dataset import Dataset
from robustness_gym.tools import recmerge, persistent_hash
from robustness_gym.identifier import Identifier


class CachedOperation:

    def __init__(self,
                 identifier: Identifier,
                 apply_fn=None,
                 ):

        # Set the identifier for the preprocessor
        self.identifier = identifier

        # Assign the apply_fn
        if apply_fn:
            self.apply = apply_fn

    @staticmethod
    def store(batch: Dict[str, List],
              updates: List[Dict]) -> Dict[str, List]:
        """
        Updates the cache of preprocessed information stored with each example in a batch.

        - batch must contain a key called 'cache' that maps to a dictionary.
        - batch['cache'] is a list of dictionaries, one per example
        - updates is a list of dictionaries, one per example
        """
        if 'cache' not in batch:
            batch['cache'] = [{} for _ in range(len(batch['index']))]

        # assert 'cache' in batch, "Examples must have a key called 'cache'."
        # assert len(batch['cache']) == len(updates), "Number of examples must equal the number of updates."

        # For each example, recursively merge the example's original cache dictionary with the update dictionary
        batch['cache'] = [recmerge(cache_dict, update_dict)
                          for cache_dict, update_dict in zip(batch['cache'], updates)]

        return batch

    def __hash__(self):
        """
        Compute a hash value for the cached operation object.
        """
        return persistent_hash(str(self.identifier))

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

        # Run the cached operation and get outputs
        processed_outputs = self.apply(*[batch[key] for key in keys])

        # Construct updates
        updates = [
            {str(self.identifier): {json.dumps(keys) if len(keys) > 1 else keys[0]: val}}
            for val in processed_outputs
        ]

        # Update the cache and return the updated batch
        return self.store(batch=batch, updates=updates)

    def apply(self, *args, **kwargs) -> List:
        """
        Implements the core functionality of the cached operation.
        """
        pass

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
