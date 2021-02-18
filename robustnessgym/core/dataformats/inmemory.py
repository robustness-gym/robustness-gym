from __future__ import annotations

import copy
import gzip
import logging
import os
import pickle
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Optional, Union

import cytoolz as tz
import datasets
import numpy as np
import pyarrow as pa
from datasets import DatasetInfo, Features, NamedSplit
from tqdm.auto import tqdm

from robustnessgym.core.dataformats.abstract import AbstractDataset
from robustnessgym.core.tools import convert_to_batch_fn

logger = logging.getLogger(__name__)

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

        # Create attributes for visible rows
        self.visible_rows = None

        # Initialization
        self._initialize_state()

        logger.info(
            f"Created `InMemoryDataset` with {len(self)} rows and "
            f"{len(self.column_names)} columns."
        )

    def _set_features(self):
        """Set the features of the dataset."""
        with self.format():
            self.info.features = Features.from_arrow_schema(
                pa.Table.from_pydict(
                    self[:1],
                ).schema
            )

    def add_column(self, column: str, values: List, overwrite=False) -> None:
        """Add a column to the dataset."""

        assert (
            column not in self.all_columns
        ) or overwrite, (
            f"Column `{column}` already exists, set `overwrite=True` to overwrite."
        )
        assert len(values) == len(self), (
            f"`add_column` failed. "
            f"Values length {len(values)} != dataset length {len(self)}."
        )

        # Add the column
        self._data[column] = list(values)
        self.all_columns.append(column)
        self.visible_columns.append(column)

        # Set features
        self._set_features()

        logging.info(f"Added column `{column}` with length `{len(values)}`.")

    def remove_column(self, column: str) -> None:
        """Remove a column from the dataset."""
        assert column in self.all_columns, f"Column `{column}` does not exist."

        # Remove the column
        del self._data[column]
        self.all_columns = [col for col in self.all_columns if col != column]
        self.visible_columns = [col for col in self.visible_columns if col != column]

        # Set features
        self._set_features()

        logging.info(f"Removed column `{column}`.")

    def select_columns(self, columns: List[str]) -> Batch:
        """Select a subset of columns."""
        for col in columns:
            assert col in self._data
        return tz.keyfilter(lambda k: k in columns, self._data)

    def _append_to_empty_dataset(self, example_or_batch: Union[Example, Batch]) -> None:
        """Append a batch of data to the dataset when it's empty."""
        # Convert to batch
        batch = self._example_or_batch_to_batch(example_or_batch)

        # TODO(karan): what other data properties need to be in sync here
        self.all_columns = self.visible_columns = list(batch.keys())

        # Dataset is empty: create the columns and append the batch
        self._data = {k: [] for k in self.column_names}
        for k in self.column_names:
            self._data[k].extend(batch[k])

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

    def _remap_index(self, index):
        if isinstance(index, int):
            return self.visible_rows[index].item()
        elif isinstance(index, slice):
            return self.visible_rows[index].tolist()
        elif isinstance(index, str):
            return index
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return self.visible_rows[index].tolist()
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            return self.visible_rows[index].tolist()
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        if (
            isinstance(index, int)
            or isinstance(index, slice)
            or isinstance(index, np.int)
        ):
            # int or slice index => standard list slicing
            return {k: self._data[k][index] for k in self.visible_columns}
        elif isinstance(index, str):
            # str index => column selection
            if index in self.column_names:
                if self.visible_rows is not None:
                    return [self._data[index][i] for i in self.visible_rows]
                return self._data[index]
            raise AttributeError(f"Column {index} does not exist.")
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return {k: [self._data[k][i] for i in index] for k in self.visible_columns}
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            return {
                k: [self._data[k][int(i)] for i in index] for k in self.visible_columns
            }
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def update(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        # input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        remove_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[InMemoryDataset]:
        """Update the columns of the dataset."""
        # TODO(karan): make this fn go faster
        # most of the time is spent on the merge, speed it up further

        # Return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return self

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return self

        # if isinstance(input_columns, str):
        #     input_columns = [input_columns]

        # Set the format
        # if input_columns is not None:
        #     previous_format = self.visible_columns
        #     self.set_format(input_columns)

        # Get some information about the function
        function_properties = self._inspect_function(function, with_indices, batched)
        assert (
            function_properties.dict_output
        ), f"`function` {function} must return dict."

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_fn(function, with_indices=with_indices)
            logger.info(f"Converting `function` {function} to batched function.")

        # Update always returns a new dataset
        logger.info("Running update, a new dataset will be returned.")
        if self.visible_rows is not None:
            # Run .map() to get updated batches and pass them into a new dataset
            new_dataset = InMemoryDataset(
                self.map(
                    (
                        lambda batch, indices: self._merge_batch_and_output(
                            batch, function(batch, indices)
                        )
                    )
                    if with_indices
                    else (
                        lambda batch: self._merge_batch_and_output(
                            batch, function(batch)
                        )
                    ),
                    with_indices=with_indices,
                    batched=True,
                    batch_size=batch_size,
                )
            )
        else:
            if function_properties.updates_existing_column:
                # Copy the ._data dict with a reference to the actual columns
                new_dataset = self.copy()

                # Calculate the values for the updated columns using a .map()
                output = self.map(
                    (
                        lambda batch, indices:
                        # Only merge columns that get updated
                        self._merge_batch_and_output(
                            {
                                k: v
                                for k, v in batch.items()
                                if k in function_properties.existing_columns_updated
                            },
                            function(batch, indices),
                        )
                    )
                    if with_indices
                    else (
                        lambda batch:
                        # Only merge columns that get updated
                        self._merge_batch_and_output(
                            {
                                k: v
                                for k, v in batch.items()
                                if k in function_properties.existing_columns_updated
                            },
                            function(batch),
                        )
                    ),
                    with_indices=with_indices,
                    batched=True,
                    batch_size=batch_size,
                )

                # Add new columns / overwrite existing columns for the update
                for col, vals in output.items():
                    new_dataset.add_column(col, vals, overwrite=True)
            else:
                # Copy the ._data dict with a reference to the actual columns
                new_dataset = self.copy()

                # Calculate the values for the new columns using a .map()
                output = new_dataset.map(
                    function=function,
                    with_indices=with_indices,
                    batched=True,
                    batch_size=batch_size,
                )
                # Add new columns for the update
                for col, vals in output.items():
                    new_dataset.add_column(col, vals)

        # # Update returns a new dataset
        # new_dataset = InMemoryDataset()
        #
        # # Run the update
        # for i, batch in enumerate(self.batch(batch_size)):
        #     # Run the function to compute the update
        #     output = (
        #         function(
        #             batch,
        #             range(i * batch_size, min(len(self), (i + 1) * batch_size)),
        #         )
        #         if with_indices
        #         else function(batch)
        #     )
        #
        #     # Merge the output with the batch
        #     output = self._merge_batch_and_output(batch, output)
        #     new_dataset.append(output)

        # # Update the features of the dataset
        # new_dataset._set_features()

        # Remove columns
        if remove_columns:
            for col in remove_columns:
                new_dataset.remove_column(col)
            logger.info(f"Removed columns {remove_columns}.")
        # Reset the format
        # if input_columns:
        #     self.set_format(previous_format)

        return new_dataset

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        **kwargs,
    ) -> Optional[Dict, List]:
        """Apply a map over the dataset."""

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        # Set the format
        previous_format = self.visible_columns
        if input_columns:
            self.set_format(input_columns)

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_fn(function, with_indices=with_indices)
            logger.info(f"Converting `function` {function} to a batched function.")

        # Run the map
        logger.info("Running `map`, the dataset will be left unchanged.")
        outputs = None
        for i, batch in tqdm(
            enumerate(self.batch(batch_size, drop_last_batch)),
            total=(len(self) // batch_size) + (1 - int(drop_last_batch)),
        ):

            # Run `function` on the batch
            output = (
                function(
                    batch,
                    range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                )
                if with_indices
                else function(batch)
            )

            if i == 0:
                # Create an empty dict or list for the outputs
                outputs = defaultdict(list) if isinstance(output, Mapping) else []

            # Append the output
            if output is not None:
                if isinstance(output, Mapping):
                    for k in output.keys():
                        outputs[k].extend(output[k])
                else:
                    outputs.extend(output)

        # Reset the format
        if input_columns:
            self.set_format(previous_format)

        if not len(outputs):
            return None
        elif isinstance(outputs, dict):
            return dict(outputs)
        return outputs

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        **kwargs,
    ) -> Optional[InMemoryDataset]:
        """Apply a filter over the dataset."""
        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        # Get some information about the function
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new dataset will be returned.")
        outputs = self.map(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
        )
        indices = np.where(outputs)[0]

        # Reset the format to set visible columns for the filter
        with self.format():
            # Filter returns a new dataset
            new_dataset = self.copy()
            new_dataset.set_visible_rows(indices)

        return new_dataset

    def copy(self, deepcopy=False):
        """Return a copy of the dataset."""
        if deepcopy:
            return copy.deepcopy(self)
        else:
            dataset = InMemoryDataset()
            dataset.__dict__ = {k: copy.copy(v) for k, v in self.__dict__.items()}
            return dataset

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {"_data", "all_columns", "visible_rows", "_info", "_split"}

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
