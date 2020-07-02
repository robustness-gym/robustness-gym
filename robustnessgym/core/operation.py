"""Implementation of the Operation abstract base class."""
import json
from abc import ABC, abstractmethod
from typing import Callable, List

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import persistent_hash


class Operation(ABC):
    """Abstract base class for operations in Robustness Gym."""

    def __init__(
        self,
        apply_fn: Callable = None,
        identifiers: List[Identifier] = None,
        num_outputs: int = None,
        *args,
        **kwargs
    ):

        if not identifiers:
            assert (
                num_outputs
            ), "Must pass in num_outputs if no identifiers are specified."

        # Set the identifiers for the outputs of the Operation
        self._identifiers = (
            Identifier.range(n=num_outputs, _name=self.__class__.__name__, **kwargs)
            if not identifiers
            else identifiers
        )

        # Assign the apply_fn
        if apply_fn:
            self.apply = apply_fn

        # # Find the batch and dataset processors
        # self._batch_processors = {method.__name__ for method in
        #                           methods_with_decorator(self.__class__,
        #                           batch_processing)}
        # self._dataset_processors = {method.__name__ for method in
        #                             methods_with_decorator(self.__class__,
        #                             dataset_processing)}

    @property
    def identifiers(self):
        return self._identifiers

    # @property
    # @abstractmethod
    # def processors(self):
    #     raise NotImplementedError("Must specify the order in which processors are
    #     applied.")
    #
    # @property
    # def batch_processors(self):
    #     return self._batch_processors
    #
    # @property
    # def dataset_processors(self):
    #     return self._dataset_processors

    def __hash__(self):
        """Compute a hash value for the cached operation object."""
        val = 0
        for identifier in self.identifiers:
            val ^= persistent_hash(str(identifier))
        return val

    # def get_cache_hash(self,
    #                    columns: List[str],
    #                    processor: str = None):
    #     """
    #     Construct a hash that will be used to identify the application of a
    #     Operation to the columns of a dataset.
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
    #     return 'cache-' + str(abs(self.get_cache_hash(columns=columns,
    #     processor=processor))) + '.arrow'

    # # FIXME: temporary
    # def __call__(self,
    #              batch_or_dataset: BatchOrDataset,
    #              columns: List[str],
    #              mask: List[int] = None,
    #              *args,
    #              **kwargs) -> BatchOrDataset:
    #
    #     if isinstance(batch_or_dataset, Dataset):
    #         # Check the Dataset's InteractionTape to see if the Operation was
    #         previously applied
    #         if not mask:
    #             # This infers a mask that specifies which outputs of the Operation
    #             are not required
    #             mask = batch_or_dataset.check_tape(
    #                 path=[self.__class__.__name__],
    #                 identifiers=self.identifiers,
    #                 columns=columns
    #             )
    #
    #         # If all outputs of the Operation were previously present in the
    #         Dataset, simply return
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
    #             f"Use Dataset.from_batch(batch) before calling {
    #             self.__class__.__name__}."
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
    #                 cache_file_name=self.get_cache_file_name(columns=columns,
    #                 processor=method)
    #             )
    #         # Apply dataset processors directly
    #         elif method.__name__ in self.dataset_processors:
    #             dataset = method(
    #                 dataset=dataset,
    #                 columns=columns,
    #             )
    #         else:
    #             raise RuntimeError(f"{method} is not a processor. "
    #                                f"Please remove {method} from the `processors`
    #                                property or decorate it.")
    #
    #     return dataset
    #
    # def process_batch(self,
    #                   batch: Batch,
    #                   columns: List[str]) -> Batch:
    #     """
    #     Apply the cached operation to a batch.
    #     """
    #     assert len(set(columns) - set(batch.keys())) == 0, "Any column in 'columns'
    #     must be present in 'batch'."
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
        """

        Args:
            obj:

        Returns:

        """
        return json.dumps(obj)

    @classmethod
    def decode(cls, s: str):
        """

        Args:
            s:

        Returns:

        """
        return json.loads(s)

    @abstractmethod
    def apply(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        pass
