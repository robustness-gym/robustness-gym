import pathlib
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Union

from robustnessgym.core.constants import CACHEDOPS
from robustnessgym.core.dataset import Batch, BatchOrDataset, Dataset
from robustnessgym.core.decorators import singlecolumn
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.operation import Operation
from robustnessgym.core.tools import (
    class_or_instancemethod,
    persistent_hash,
    recmerge,
    strings_as_json,
)


class CachedOperation(Operation):
    """Class to create CachedOperations."""

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / "robustnessgym/cachedops/"

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    def __init__(
        self, apply_fn: Callable = None, identifier: Identifier = None, *args, **kwargs
    ):

        super(CachedOperation, self).__init__(
            apply_fn=apply_fn,
            identifiers=[identifier] if identifier else None,
            num_outputs=1,
            *args,
            **kwargs,
        )

    def __repr__(self):
        """Representation of a cached operation object.

        Returns: string representation
        """
        return str(self.identifier)

    @property
    def identifier(self):
        return self.identifiers[0]

    @staticmethod
    def store(batch: Batch, updates: List[Dict]) -> Batch:
        """Updates the cache of preprocessed information stored with each
        example in a batch.

        Args:
            batch: a batch of data
            updates: a list of dictionaries, one per example

        Returns: updated batch
        """
        if "cache" not in batch:
            batch["cache"] = [{} for _ in range(len(batch["index"]))]

        # For each example, recursively merge the example's original cache dictionary
        # with the update dictionary
        batch["cache"] = [
            recmerge(cache_dict, update_dict)
            for cache_dict, update_dict in zip(batch["cache"], updates)
        ]

        return batch

    @class_or_instancemethod
    def retrieve(
        self_or_cls,
        batch: Batch,
        columns: Union[List[str], List[List[str]]],
        proc_fns: Union[str, Callable, List[Union[str, Callable]]] = None,
        identifier: Union[str, Identifier] = None,
        reapply: bool = False,
        **kwargs,
    ) -> Optional[Union[Dict[tuple, List], List[List], Batch, List[Batch]]]:
        """Retrieve information from the cache.

        Args:
            batch: a batch of data
            columns: list of columns to retrieve cached information for
            proc_fns: list of processing functions to be executed left to right on
            the cached data
            identifier: name of the identifier to retrieve
            reapply: whether to recompute the cached operation at retrieval

        Returns: dict mapping a column to a list of length len(batch)
        """
        if not reapply:
            # Infer the most relevant key to retrieve if an identifier is not specified
            if not identifier:
                if isinstance(self_or_cls, type):
                    # cls
                    target_ident_key = self_or_cls.__name__
                else:
                    # self
                    target_ident_key = str(self_or_cls.identifier)

                cachedop_columns = defaultdict(list)
                for ident_key in batch.keys():
                    # Parse the identifier
                    ident = Identifier.parse(ident_key)
                    cachedop_columns[ident.without("columns")].append(ident)

                best_match, best_distance = None, 100000000
                for ident in cachedop_columns:
                    ident_key = str(ident)
                    # Pick the key that best matches the cls name or instance identifier
                    if (
                        ident_key.startswith(target_ident_key)
                        and len(ident_key.replace(target_ident_key, "")) < best_distance
                    ):
                        best_match = ident
                        best_distance = len(ident_key.replace(target_ident_key, ""))

                identifier = best_match

                # Still no identifier
                if not identifier:
                    raise ValueError(
                        f"Retrieval failed: couldn't find a key called "
                        f"{target_ident_key} in cache."
                    )

            try:
                if isinstance(columns, str) or (
                    isinstance(columns[0], str) and len(columns) == 1
                ):
                    if isinstance(columns, str):
                        columns = [columns]
                    # Retrieving single piece of information for a single column
                    retrieval = [
                        self_or_cls.decode(val)
                        for val in batch[identifier(columns=columns)]
                    ]

                elif isinstance(columns[0], str):
                    # Retrieving single piece of information `columns` list
                    retrieval = {
                        tuple(columns): [
                            self_or_cls.decode(val)
                            for val in batch[identifier(columns=columns)]
                        ]
                    }
                else:
                    # Retrieving multiple pieces of information
                    retrieval = {
                        tuple(cols_)
                        if len(cols_) > 1
                        else cols_[0]: [
                            self_or_cls.decode(val)
                            for val in batch[identifier(columns=cols_)]
                        ]
                        for cols_ in columns
                    }

            except KeyError:
                raise KeyError(
                    "Could not retrieve information for all columns. "
                    "If you're trying to retrieve information for multiple columns, "
                    "use columns=[[col_1], [col_2], ..] "
                    "instead of columns=[col_1, col_2, ..]."
                )

            # Check if the retrieved information needs to be processed
            if not proc_fns:
                return retrieval

            # Resolve the str proc_fns to callable(s)
            if isinstance(proc_fns, str):
                proc_fns = getattr(self_or_cls, proc_fns)
            elif isinstance(proc_fns, List):
                proc_fns = [
                    proc_fn
                    if isinstance(proc_fn, Callable)
                    else getattr(self_or_cls, proc_fn)
                    for proc_fn in proc_fns
                ]

            # Process and return the retrieved information
            if isinstance(proc_fns, Callable):
                if isinstance(retrieval, list):
                    return proc_fns(retrieval)
                else:
                    return {k: proc_fns(v) for k, v in retrieval.items()}

            if isinstance(retrieval, list):
                return [[proc_fn(v) for v in retrieval] for proc_fn in proc_fns]

            return [
                {k: proc_fn(v) for k, v in retrieval.items()} for proc_fn in proc_fns
            ]

        else:
            if proc_fns:
                print("Warning: proc_fns has no effect when reapply=True.")

            # Run the operation on the fly
            # TODO(karan): does this work for ops that require process_dataset
            if isinstance(columns, str) or (
                isinstance(columns[0], str) and len(columns) == 1
            ):
                if isinstance(columns, str):
                    columns = [columns]
                return (
                    self_or_cls(**kwargs).apply(batch=batch, columns=columns)
                    if isinstance(self_or_cls, type)
                    else self_or_cls.apply(batch=batch, columns=columns)
                )
            elif isinstance(columns[0], str):
                return {
                    tuple(columns): self_or_cls(**kwargs).apply(
                        batch=batch, columns=columns
                    )
                    if isinstance(self_or_cls, type)
                    else self_or_cls.apply(batch=batch, columns=columns)
                }
            return {
                tuple(cols_)
                if len(cols_) > 1
                else cols_[0]: self_or_cls(**kwargs).apply(batch=batch, columns=cols_)
                if isinstance(self_or_cls, type)
                else self_or_cls.apply(batch=batch, columns=cols_)
                for cols_ in columns
            }

    # @class_or_instancemethod
    # def retrieve(
    #         self_or_cls,
    #         batch: Batch,
    #         columns: Union[List[str], List[List[str]]],
    #         proc_fns: Union[str, Callable, List[Union[str, Callable]]] = None,
    #         identifier: Union[str, Identifier] = None,
    #         reapply: bool = False,
    #         **kwargs,
    # ) -> Optional[Union[Batch, List[Batch]]]:
    #     """Retrieve information from the cache.
    #
    #     Args:
    #         batch: a batch of data
    #         columns: list of columns to retrieve cached information for
    #         proc_fns: list of processing functions to be executed left to right on
    #         the cached data
    #         identifier: name of the identifier to retrieve
    #         reapply: whether to recompute the cached operation at retrieval
    #
    #     Returns: dict mapping a column to a list of length len(batch)
    #     """
    #     if not reapply:
    #         # Nothing to return if there's no cache
    #         if "cache" not in batch:
    #             raise ValueError(
    #                 "`cache` key missing: nothing has been cached yet. "
    #                 "Are you sure you ran a CachedOperation?"
    #             )
    #
    #         # Infer the most relevant key to retrieve if an identifier is not
    #         # specified
    #         if not identifier:
    #             if isinstance(self_or_cls, type):
    #                 # cls
    #                 target_ident_key = self_or_cls.__name__
    #             else:
    #                 # self
    #                 target_ident_key = str(self_or_cls.identifier)
    #
    #             best_match, best_distance = None, 100000000
    #             for ident_key in batch["cache"][0].keys():
    #                 # Pick the key that best matches the cls name or instance
    #                 # identifier
    #                 if (
    #                         ident_key.startswith(target_ident_key)
    #                         and len(
    #                     ident_key.replace(target_ident_key, "")) < best_distance
    #                 ):
    #                     best_match = ident_key
    #                     best_distance = len(ident_key.replace(target_ident_key, ""))
    #
    #             identifier = best_match
    #
    #             # Still no identifier
    #             if not identifier:
    #                 raise ValueError(
    #                     f"Retrieval failed: couldn't find a key called "
    #                     f"{target_ident_key} in cache."
    #                 )
    #
    #         try:
    #             if isinstance(columns[0], str):
    #                 retrieval = {
    #                     strings_as_json(columns): [
    #                         self_or_cls.decode(
    #                             cache[str(identifier)][strings_as_json(columns)]
    #                         )
    #                         for cache in batch["cache"]
    #                     ]
    #                 }
    #             else:
    #                 retrieval = {
    #                     strings_as_json(cols_): [
    #                         self_or_cls.decode(
    #                             cache[str(identifier)][strings_as_json(cols_)]
    #                         )
    #                         for cache in batch["cache"]
    #                     ]
    #                     for cols_ in columns
    #                 }
    #
    #         except KeyError:
    #             raise KeyError(
    #                 "Could not retrieve information for all columns. "
    #                 "If you're trying to retrieve information for multiple columns, "
    #                 "use columns=[[col_1], [col_2], ..] "
    #                 "instead of columns=[col_1, col_2, ..]."
    #             )
    #
    #         # Check if the retrieved information needs to be processed
    #         if not proc_fns:
    #             return retrieval
    #
    #         # Resolve the str proc_fns to callable(s)
    #         if isinstance(proc_fns, str):
    #             proc_fns = getattr(self_or_cls, proc_fns)
    #         elif isinstance(proc_fns, List):
    #             proc_fns = [
    #                 proc_fn
    #                 if isinstance(proc_fn, Callable)
    #                 else getattr(self_or_cls, proc_fn)
    #                 for proc_fn in proc_fns
    #             ]
    #
    #         # Process and return the retrieved information
    #         if isinstance(proc_fns, Callable):
    #             return {k: proc_fns(v) for k, v in retrieval.items()}
    #
    #         return [
    #             {k: proc_fn(v) for k, v in retrieval.items()} for proc_fn in proc_fns
    #         ]
    #
    #     else:
    #         if proc_fns:
    #             print("Warning: proc_fns has no effect when reapply=True.")
    #
    #         # Run the operation on the fly
    #         # TODO(karan): does this work for ops that require process_dataset
    #         if isinstance(columns[0], str):
    #             return {
    #                 strings_as_json(columns): self_or_cls(**kwargs).apply(
    #                     batch=batch, columns=columns
    #                 )
    #                 if isinstance(self_or_cls, type)
    #                 else self_or_cls.apply(batch=batch, columns=columns)
    #             }
    #         return {
    #             strings_as_json(cols_): self_or_cls(**kwargs).apply(
    #                 batch=batch, columns=cols_
    #             )
    #             if isinstance(self_or_cls, type)
    #             else self_or_cls.apply(batch=batch, columns=cols_)
    #             for cols_ in columns
    #         }

    @classmethod
    def available(cls, batch: Batch):
        """Check if the cached operation is available to retrieve in
        `batch`."""
        return any([key.startswith(cls.__name__) for key in batch.keys()])

    def get_cache_hash(self, columns: Optional[List[str]] = None):
        """Construct a hash that will be used to identify the application of a
        cached operation to the columns of a dataset."""

        val = hash(self)
        if columns:
            for key in columns:
                val ^= persistent_hash(key)
        return val

    def get_cache_file_name(self, columns=None):
        """Construct a file name for caching."""
        return "cache-" + str(abs(self.get_cache_hash(columns=columns))) + ".arrow"

    def prepare_batch(
        self,
        batch: Batch,
        columns: List[str],
        *args,
        **kwargs,
    ) -> Batch:
        """Preparation that is applied before the CachedOperation.

        This is provided as a convenience function that can be called by
        prepare_dataset.

        Args:
            batch: batch of examples
            columns: list of columns

        Returns: updated batch
        """
        raise NotImplementedError("Implement `prepare_batch`.")

    def prepare_dataset(
        self,
        dataset: Dataset,
        columns: List[str],
        batch_size: int = 32,
        *args,
        **kwargs,
    ) -> None:
        """Preparation that is applied before the CachedOperation.

        Many CachedOperations require a full pass over the dataset to precompute some
        variables before the core operation can actually be applied e.g. to create a
        Bag-of-Words representation, constructing a dataset vocabulary to keep only
        tokens that are frequently seen across the dataset.

        Args:
            dataset: Dataset
            columns: list of columns
            batch_size: batch size for .map(..)

        Returns: updated Dataset
        """

        # Set the data format
        dataset.set_format(columns)

        # Batch the dataset, and prepare each batch
        for batch in dataset.batch(batch_size):
            try:
                # Check if the `prepare_batch` function has been implemented
                self.prepare_batch(
                    batch=batch,
                    columns=columns,
                    *args,
                    **kwargs,
                )
            except NotImplementedError:
                break

        # Reset the data format
        dataset.reset_format()

        # # Apply preparation to the dataset
        # # TODO(karan): this is similar to the try except block for slicebuilders,
        # #  refactor
        # try:
        #     return dataset.map(
        #         partial(self.prepare_batch, columns=columns),
        #         batched=True,
        #         batch_size=batch_size,
        #         # The cache file name is a XOR of the interaction history and the
        #         # current operation
        #         # FIXME(karan): this is repeated
        #         cache_file_name=str(
        #             dataset.logdir
        #             / (
        #                 "cache-"
        #                 + str(
        #                     abs(
        #                         persistent_hash(str(dataset.identifier))
        #                         ^ dataset.hash_interactions()
        #                         ^ persistent_hash(
        #                             str(self.identifier) +
        #                             str(strings_as_json(columns))
        #                         )
        #                     )
        #                 )
        #                 + "-prep.arrow"
        #             )
        #         ),
        #     )
        # except:  # TypeError or PicklingError or AttributeError: # noqa
        #     # Batch the dataset, and process each batch
        #     all_batches = [
        #         self.prepare_batch(
        #             batch=batch,
        #             columns=columns,
        #         )
        #         for batch in dataset.batch(batch_size)
        #     ]
        #
        #     return dataset.map(
        #         lambda examples, indices: all_batches[indices[0] // batch_size],
        #         batched=True,
        #         batch_size=batch_size,
        #         with_indices=True,
        #         load_from_cache_file=False,
        #         # The cache file name is a XOR of the interaction history and the
        #         # current operation
        #         # FIXME(karan): this is repeated
        #         cache_file_name=str(
        #             dataset.logdir
        #             / (
        #                 "cache-"
        #                 + str(
        #                     abs(
        #                         persistent_hash(str(dataset.identifier))
        #                         ^ dataset.hash_interactions()
        #                         ^ persistent_hash(
        #                             str(self.identifier) +
        #                             str(strings_as_json(columns))
        #                         )
        #                     )
        #                 )
        #                 + "-prep.arrow"
        #             )
        #         ),
        #     )

    def apply(self, batch: Batch, columns: List[str], *args, **kwargs) -> List:
        """Implements the core functionality of the cached operation."""
        pass

    def _construct_updates(self, encoded_outputs: List[str], columns: List[str]):
        return [
            {str(self.identifier): {strings_as_json(columns): val}}
            for val in encoded_outputs
        ]

    def process_batch(
        self,
        batch: Batch,
        columns: List[str],
        *args,
        **kwargs,
    ) -> Batch:
        """Apply the cached operation to a batch."""
        assert len(set(columns) - set(batch.keys())) == 0, (
            f"All `columns` ({columns}) must be present in `batch` ("
            f"{list(batch.keys())})."
        )

        # Run the cached operation, and encode outputs (defaults to json.dumps)
        encoded_outputs = [
            self.encode(example_output)
            for example_output in self.apply(batch=batch, columns=columns)
        ]

        return {str(self.identifier(columns=columns)): encoded_outputs}

        # # Update the batch and return the updated batch
        # batch[str(self.identifier(columns=columns))] = encoded_outputs
        #
        # return batch

        # # Construct updates
        # updates = self._construct_updates(
        #     encoded_outputs=encoded_outputs, columns=columns
        # )
        #
        # # Update the cache and return the updated batch
        # return self.store(batch=batch, updates=updates)

    def process_dataset(
        self,
        dataset: Dataset,
        columns: List[str],
        batch_size: int = 32,
    ) -> Dataset:
        """Apply the cached operation to a dataset."""

        return dataset.update(
            partial(self.process_batch, columns=columns),
            batched=True,
            batch_size=batch_size,
        )

        # try:
        #     return dataset.map(
        #         partial(self.process_batch, columns=columns),
        #         batched=True,
        #         batch_size=batch_size,
        #         # The cache file name is a XOR of the interaction history and the
        #         # current operation
        #         cache_file_name=str(
        #             dataset.logdir
        #             / (
        #                 "cache-"
        #                 + str(
        #                     abs(
        #                         persistent_hash(str(dataset.identifier))
        #                         ^ dataset.hash_interactions()
        #                         ^ persistent_hash(
        #                             str(self.identifier) +
        #                             str(strings_as_json(columns))
        #                         )
        #                     )
        #                 )
        #                 + ".arrow"
        #             )
        #         ),
        #         # self.get_cache_file_name(columns=columns),
        #     )
        # except:  # noqa
        #     # Batch the dataset, and process each batch
        #     all_batches = [
        #         self.process_batch(
        #             batch=batch,
        #             columns=columns,
        #         )
        #         for batch in dataset.batch(batch_size)
        #     ]
        #
        #     return dataset.map(
        #         lambda examples, indices: all_batches[indices[0] // batch_size],
        #         batched=True,
        #         batch_size=batch_size,
        #         with_indices=True,
        #         load_from_cache_file=False,
        #         # The cache file name is a XOR of the interaction history and the
        #         # current operation
        #         cache_file_name=str(
        #             dataset.logdir
        #             / (
        #                 "cache-"
        #                 + str(
        #                     abs(
        #                         persistent_hash(str(dataset.identifier))
        #                         ^ dataset.hash_interactions()
        #                         ^ persistent_hash(
        #                             str(self.identifier) +
        #                             str(strings_as_json(columns))
        #                         )
        #                     )
        #                 )
        #                 + ".arrow"
        #             )
        #         ),
        #     )

    def __call__(
        self, batch_or_dataset: BatchOrDataset, columns: List[str], batch_size: int = 32
    ) -> BatchOrDataset:

        if isinstance(batch_or_dataset, Dataset):

            # Check the InteractionTape to see if the CachedOperation was applied
            if batch_or_dataset.check_tape(
                path=[CACHEDOPS],
                identifiers=self.identifier,
                columns=columns,
            ):
                return batch_or_dataset

            # Prepare to apply the CachedOperation to the dataset
            self.prepare_dataset(
                dataset=batch_or_dataset,
                columns=columns,
                batch_size=batch_size,
            )

            # Apply the CachedOperation to the dataset
            dataset = self.process_dataset(
                dataset=batch_or_dataset,
                columns=columns,
                batch_size=batch_size,
            )

            # Update the InteractionTape with the applied CachedOperation
            dataset.update_tape(
                path=[CACHEDOPS],
                identifiers=self.identifier,
                columns=columns,
            )

            return dataset

        elif isinstance(batch_or_dataset, Dict):
            # Apply the CachedOperation
            updates = self.process_batch(batch=batch_or_dataset, columns=columns)
            # Merge assuming that updates doesn't affect existing columns
            assert len(set(updates.keys()).intersection(batch_or_dataset.keys())) == 0
            return {**batch_or_dataset, **updates}
        else:
            raise NotImplementedError


class SingleColumnCachedOperation(CachedOperation):
    def __call__(
        self, batch_or_dataset: BatchOrDataset, columns: List[str], batch_size: int = 32
    ) -> BatchOrDataset:
        """Apply independently to each column.

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
    def apply(self, batch: Batch, columns: List[str], *args, **kwargs) -> List:
        return self.single_column_apply(batch[columns[0]])

    def single_column_apply(self, column_batch: List, **kwargs) -> List:
        raise NotImplementedError("Must implement single_column_apply.")


class ScoreOperation(CachedOperation):
    def apply(
        self, batch: Batch, columns: List[str], *args, **kwargs
    ) -> List[Union[int, float]]:
        return super().apply(batch, columns, *args, **kwargs)


def stow(
    dataset: Dataset,
    cached_ops: Dict[CachedOperation, List[List[str]]],
    batch_size: int = 32,
    load_from_cache_file: bool = True,
):
    """Apply a list of cached operations in sequence."""

    # Check the InteractionTape to remove CachedOperations that have already been stowed
    for cached_op, list_of_columns in list(cached_ops.items()):
        indices_to_remove = []
        for i, columns in enumerate(list(list_of_columns)):
            if dataset.check_tape(
                path=[CACHEDOPS],
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
    #     Consolidate the application of the CachedOperations passed to stow into a
    #     single mappable function.
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
    #         "-".join(str(k) + "-" + str(v) for k, v in f.items()) for f in
    #         dataset._data_files
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
                path=[CACHEDOPS], identifiers=cached_op.identifier, columns=columns
            )

    return dataset
