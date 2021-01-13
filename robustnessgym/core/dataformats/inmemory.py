from __future__ import annotations

import gzip
import os
import pickle
from types import SimpleNamespace
from typing import Callable, Dict, List, Mapping, Optional, Union

import cytoolz as tz
import datasets
import pyarrow as pa
from datasets import DatasetInfo, Features, NamedSplit

from robustnessgym.core.dataformats.abstract import AbstractDataset
from robustnessgym.core.tools import recmerge

Example = Dict
Batch = Dict[str, List]


class InMemoryDataset(AbstractDataset):
    """Class for datasets that are to be stored in memory."""

    def __init__(
        self,
        *args,
        column_names: List[str] = None,
        info: DatasetInfo = None,
        split: Optional[NamedSplit] = None,
    ):

        # Data is a dictionary of lists
        self._data = {}

        # Single argument
        if len(args) == 1:
            assert column_names is None, "Don't pass in column_names."
            # The data is passed in
            data = args[0]

            # `data` is a dictionary
            if isinstance(data, dict) and len(data):
                # Assert all columns are the same length
                self._assert_columns_all_equal_length(data)
                self._data = data

            # `data` is a list
            elif isinstance(data, list) and len(data):
                # Transpose the list of dicts to a dict of lists i.e. a batch
                data = tz.merge_with(list, *data)
                # Assert all columns are the same length
                self._assert_columns_all_equal_length(data)
                self._data = data

            # `data` is a datasets.Dataset
            elif isinstance(data, datasets.Dataset):
                self._data = data[:]
                info, split = data.info, data.split

        # No argument
        elif len(args) == 0:

            # Use column_names to setup the data dictionary
            if column_names:
                self._data = {k: [] for k in column_names}

        # Setup the DatasetInfo
        info = info.copy() if info is not None else DatasetInfo()
        AbstractDataset.__init__(self, info=info, split=split)

        # Create attributes for all columns and visible columns
        self.all_columns = list(self._data.keys())
        self.visible_columns = None

        # Initialization
        self._initialize_state()

    def _initialize_state(self):
        """Dataset state initialization."""
        # Show all columns by default
        self.visible_columns = self.all_columns

        # Set the features
        self._set_features()

    def _set_features(self):
        """Set the features of the dataset."""
        self.info.features = Features.from_arrow_schema(
            pa.Table.from_pydict(
                self[:1],
            ).schema
        )

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def _assert_columns_all_equal_length(cls, batch: Batch):
        """Check that all columns have the same length so that the data is
        tabular."""
        assert cls._columns_all_equal_length(
            batch
        ), "All columns must have equal length."

    @classmethod
    def _columns_all_equal_length(cls, batch: Batch):
        """Check that all columns have the same length so that the data is
        tabular."""
        if len(set([len(v) for k, v in batch.items()])) == 1:
            return True
        return False

    def _check_columns_exist(self, columns: List[str]):
        """Check that every column in `columns` exists."""
        for col in columns:
            assert col in self.all_columns, f"{col} is not a valid column."

    @property
    def column_names(self):
        """Column names in the dataset."""
        return self.all_columns

    @property
    def columns(self):
        """Column names in the dataset."""
        return self.column_names

    @property
    def num_rows(self):
        """Number of rows in the dataset."""
        return len(self)

    def set_format(self, columns: List[str]):
        """Set the dataset format.

        Only `columns` are visible after set_format is invoked.
        """
        # Check that the columns exist
        self._check_columns_exist(columns)

        # Set visible columns
        self.visible_columns = columns

    def reset_format(self):
        """Reset the dataset format.

        All columns are visible.
        """
        # All columns are visible
        self.visible_columns = self.all_columns
        # TODO(karan): all_columns should be updated if a column is added by map

    def batch(self, batch_size: int = 32, drop_last_batch: bool = False):
        """Batch the dataset.

        Args:
            batch_size: integer batch size
            drop_last_batch: drop the last batch if its smaller than batch_size

        Returns:
            batches of data
        """
        for i in range(0, len(self), batch_size):
            if drop_last_batch and i + batch_size > len(self):
                continue
            yield self[i : i + batch_size]

    def _example_or_batch_to_batch(
        self, example_or_batch: Union[Example, Batch]
    ) -> Batch:

        # Check if example_or_batch is a batch
        is_batch = all(
            [isinstance(v, List) for v in example_or_batch.values()]
        ) and self._columns_all_equal_length(example_or_batch)

        # Convert to a batch if not
        if not is_batch:
            batch = {k: [v] for k, v in example_or_batch.items()}
        else:
            batch = example_or_batch

        return batch

    def _append_to_empty_dataset(self, example_or_batch: Union[Example, Batch]) -> None:
        """Append a batch of data to the dataset when it's empty."""
        # Convert to batch
        batch = self._example_or_batch_to_batch(example_or_batch)

        # Dataset is empty: just assign it to the batch
        self._data = batch
        # TODO(karan): what other data properties need to be in sync here
        self.all_columns = self.visible_columns = list(self._data.keys())

    def append(
        self,
        example_or_batch: Union[Example, Batch],
    ) -> None:
        """Append a batch of data to the dataset.

        `batch` must have the same columns as the dataset (regardless of
        what columns are visible).
        """
        if not self.column_names:
            return self._append_to_empty_dataset(example_or_batch)

        # Check that example_or_batch has the same format as the dataset
        # TODO(karan): require matching on nested features?
        columns = list(example_or_batch.keys())
        assert set(columns) == set(
            self.column_names
        ), f"Mismatched columns\nbatch: {columns}\ndataset: {self.column_names}"

        # Convert to a batch
        batch = self._example_or_batch_to_batch(example_or_batch)

        # Append to the dataset
        for k in self.column_names:
            self._data[k].extend(batch[k])

    def __len__(self):
        # Pick any column
        column_names = self.column_names
        if column_names:
            return len(self._data[column_names[0]])
        return 0

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, slice):
            return {k: self._data[k][index] for k in self.visible_columns}
        elif isinstance(index, tuple):
            raise NotImplementedError("Tuple as index")
        elif isinstance(index, str):
            if index in self.column_names:
                return self._data[index]
            raise AttributeError(f"Column {index} does not exist.")
        elif isinstance(index, list) and len(index):
            return {k: [self._data[k][i] for i in index] for k in self.visible_columns}
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    @classmethod
    def from_batch(cls, batch: Batch):
        """Create an InMemoryDataset from a batch."""
        return cls(batch)

    @classmethod
    def from_batches(cls, batches: List[Batch]):
        """Create an InMemoryDataset from a list of batches."""
        return cls.from_batch(
            tz.merge_with(tz.concat, *batches),
        )

    def select_columns(self, columns: List[str]) -> Batch:
        """Select a subset of columns."""
        for col in columns:
            assert col in self._data
        return tz.keyfilter(lambda k: k in columns, self._data)

    def _inspect_function(
        self,
        function: Callable,
        with_indices: bool = False,
        batched: bool = False,
    ) -> SimpleNamespace:

        # Initialize
        no_output = dict_output = bool_output = False

        # Run the function to test it
        if batched:
            if with_indices:
                output = function(self[:2], range(2))
            else:
                output = function(self[:2])

        else:
            if with_indices:
                output = function(self[0], 0)
            else:
                output = function(self[0])

        if isinstance(output, Mapping):
            dict_output = True
        elif output is None:
            no_output = True
        elif isinstance(output, bool):
            bool_output = True
        else:
            raise ValueError("function must return a dict or None.")

        return SimpleNamespace(
            dict_output=dict_output,
            no_output=no_output,
            bool_output=bool_output,
        )

    @classmethod
    def _merge_batch_and_output(cls, batch: Batch, output: Batch):
        """Merge an output during .map() into a batch."""
        combined = batch
        for k in output.keys():
            if k not in batch:
                combined[k] = output[k]
            else:
                if isinstance(batch[k][0], dict) and isinstance(output[k][0], dict):
                    combined[k] = [
                        recmerge(b_i, o_i) for b_i, o_i in zip(batch[k], output[k])
                    ]
                else:
                    combined[k] = output[k]
        return combined

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        # keep_in_memory: bool = False,
        #             # load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        #             # writer_batch_size: Optional[int] = 1000,
        #             # features: Optional[Features] = None,
        #             # disable_nullable: bool = False,
        #             # fn_kwargs: Optional[dict] = None,
        #             # num_proc: Optional[int] = None,
        #             # suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        #             # new_fingerprint: Optional[str] = None,
        required_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[InMemoryDataset]:

        # Set the function if it's None
        if function is None:
            function = (lambda x, index: x) if with_indices else lambda x: x

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        # Set the format
        if input_columns:
            previous_format = self.visible_columns
            self.set_format(input_columns)

        # Get some information about the function
        function_properties = self._inspect_function(function, with_indices, batched)
        update_dataset = function_properties.dict_output

        # Return if `self` has no examples
        if not len(self):
            return self if update_dataset else None

        # Map returns a new dataset if the function returns a dict
        if update_dataset:
            new_dataset = InMemoryDataset()

        # Run the map
        if batched:
            for i, batch in enumerate(self.batch(batch_size, drop_last_batch)):
                output = (
                    function(
                        batch,
                        range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                    )
                    if with_indices
                    else function(batch)
                )

                if update_dataset:
                    # TODO(karan): check that this has the correct behavior
                    output = self._merge_batch_and_output(batch, output)
                    # output = recmerge(batch, output, merge_sequences=True)
                    new_dataset.append(output)

        else:
            for i, example in enumerate(self):
                output = function(example, i) if with_indices else function(example)

                if update_dataset:
                    # TODO(karan): check that this has the correct behavior
                    output = recmerge(example, output)
                    new_dataset.append(output)

        # Reset the format
        if input_columns:
            self.set_format(previous_format)

        if update_dataset:
            new_dataset._set_features()
            return new_dataset
        return None

    @classmethod
    def _mask_batch(cls, batch: Batch, boolean_mask: List[bool]):
        """Remove elements in `batch` that are masked by `boolean_mask`."""
        return {
            k: [e for i, e in enumerate(v) if boolean_mask[i]] for k, v in batch.items()
        }

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batch_size: Optional[int] = 1000,
        remove_columns: Optional[List[str]] = None,
        # keep_in_memory: bool = False,
        # load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        # writer_batch_size: Optional[int] = 1000,
        # fn_kwargs: Optional[dict] = None,
        # num_proc: Optional[int] = None,
        # suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        # new_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        # Set the function if it's None
        if function is None:
            function = lambda *args, **kwargs: True

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        # Set the format
        if input_columns:
            previous_format = self.visible_columns
            self.set_format(input_columns)

        # Get some information about the function
        # TODO(karan): extend to handle batched functions
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=False,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Run the filter
        indices = []
        for i, example in enumerate(self):
            output = (
                function(
                    example,
                    i,
                )
                if with_indices
                else function(example)
            )
            assert isinstance(output, bool), "function must return boolean."

            # Add in the index
            if output:
                indices.append(i)

        # Reset the format, to set visible columns for the filter
        self.reset_format()

        # Filter returns a new dataset
        new_dataset = InMemoryDataset()
        if indices:
            new_dataset = InMemoryDataset.from_batch(self[indices])

        # Set the format back to what it was before the filter was applied
        if input_columns:
            self.set_format(previous_format)

        return new_dataset

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {"_data", "all_columns", "_info", "_split"}

    @classmethod
    def _assert_state_keys(cls, state: Dict) -> None:
        """Assert that a state contains all required keys."""
        assert (
            set(state.keys()) == cls._state_keys()
        ), f"State must contain all state keys: {cls._state_keys()}."

    def __getstate__(self) -> Dict:
        """Get the internal state of the dataset."""
        state = {key: getattr(self, key) for key in self._state_keys()}
        self._assert_state_keys(state)
        return state

    def __setstate__(self, state: Dict) -> None:
        """Set the internal state of the dataset."""
        if not isinstance(state, dict):
            raise ValueError(
                f"`state` must be a dictionary containing " f"{self._state_keys()}."
            )

        self._assert_state_keys(state)

        for key in self._state_keys():
            setattr(self, key, state[key])

        # Do some initialization
        self._initialize_state()

    @classmethod
    def load_from_disk(cls, path: str) -> InMemoryDataset:
        """Load the in-memory dataset from disk."""

        with gzip.open(os.path.join(path, "data.gz")) as f:
            dataset = pickle.load(f)
        # # Empty state dict
        # state = {}
        #
        # # Load the data
        # with gzip.open(os.path.join(path, "data.gz")) as f:
        #     state['_data'] = pickle.load(f)
        #
        # # Load the metadata
        # metadata = json.load(
        #     open(os.path.join(path, "metadata.json"))
        # )
        #
        # # Merge the metadata into the state
        # state = {**state, **metadata}

        # Create an empty `InMemoryDataset` and set its state
        # dataset = cls()
        # dataset.__setstate__(state)

        return dataset

    def save_to_disk(self, path: str):
        """Save the in-memory dataset to disk."""
        # Create all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Store the data in a compressed format
        with gzip.open(os.path.join(path, "data.gz"), "wb") as f:
            pickle.dump(self, f)

        # # Get the dataset state
        # state = self.__getstate__()
        #
        # # Store the data in a compressed format
        # with gzip.open(os.path.join(path, "data.gz"), "wb") as f:
        #     pickle.dump(state['_data'], f)
        #
        # # Store the metadata
        # json.dump(
        #     {k: v for k, v in state.items() if k != '_data'},
        #     open(os.path.join(path, "metadata.json"), 'w'),
        # )
