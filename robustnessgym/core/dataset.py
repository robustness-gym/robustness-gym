from __future__ import annotations

import json
import logging
import os
import pathlib
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import Callable, Dict, List, Optional, Sequence, Union

import cytoolz as tz
import datasets
import pandas as pd
from datasets import Features
from pyarrow import json as jsonarrow
from pyarrow import table

from robustnessgym.core.dataformats.inmemory import InMemoryDataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tape import InteractionTapeHierarchyMixin

logger = logging.getLogger(__name__)

Batch = Dict[str, List]
BatchOrDataset = Union[Batch, "Dataset"]


class Dataset(
    InteractionTapeHierarchyMixin,
):
    """RobustnessGym Dataset class."""

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / "robustnessgym/datasets/"

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    def __init__(
        self,
        *args,
        identifier: Identifier = None,
        dataset_fmt: str = None,
        **kwargs,
    ):

        logger.debug("Creating Dataset.")

        # Internally, keep track of data inside a dataset
        self._dataset = None

        # Set a default dataset_fmt
        if len(args) == 1 and isinstance(args[0], datasets.Dataset):
            # If no `dataset_fmt` is passed in with a datasets.Dataset, just assume
            # `dataset_fmt` should be 'datasets'
            dataset_fmt = "datasets" if dataset_fmt is None else dataset_fmt
        else:
            # Default to `in_memory`, unless a dataset_fmt is passed in explicitly
            dataset_fmt = "in_memory" if dataset_fmt is None else dataset_fmt

        # InMemoryDataset
        if dataset_fmt == "in_memory":
            if len(args) == 1 and isinstance(args[0], InMemoryDataset):
                # Assign the dataset directly
                self._dataset = args[0]
            else:
                # Assign the dataset after converting to an InMemoryDataset
                self._dataset = InMemoryDataset(*args, **kwargs)

        # datasets.Dataset
        elif dataset_fmt == "datasets":
            if len(args) == 1 and isinstance(args[0], datasets.Dataset):
                # Assign the dataset directly
                self._dataset = args[0]
            else:
                # Assign the dataset after converting to a datasets.Dataset
                self._dataset = datasets.Dataset(*args, **kwargs)

        else:
            raise NotImplementedError(
                "`dataset_fmt` must be one of ['in_memory', 'datasets']."
            )

        # Store the dataset format
        self._dataset_fmt = dataset_fmt

        # Call the InteractionTapeHierarchyMixin constructor
        InteractionTapeHierarchyMixin.__init__(self)

        # Create an identifier
        self._identifier = (
            self._autobuild_identifier() if not identifier else identifier
        )

        # Create logging directory
        self._create_logdir()

        # Add an index to the dataset
        if not self.has_index:
            self._add_index()

    @property
    def identifier(self):
        """Identifier."""
        return self._identifier

    @property
    def dataset_fmt(self):
        """Dataset format, one of ['in_memory', 'datasets']."""
        return self._dataset_fmt

    @property
    def features(self):
        """Dataset features."""
        return self._dataset.features

    @property
    def info(self):
        """Dataset info."""
        return self._dataset.info

    @property
    def split(self):
        """Dataset split."""
        return self._dataset.split

    @property
    def num_rows(self):
        """Number of rows in the dataset."""
        return self._dataset.num_rows

    @contextmanager
    def format(self, columns: List[str] = None):
        """Context where only `columns` will be visible."""
        # Get the current format
        current_format = self.get_format()

        if columns:
            # View only `columns`
            self.set_format(columns)
        else:
            # Use all columns
            self.set_format(self.column_names)
        try:
            yield
        finally:
            # Reset the format back
            self.set_format(current_format)

    def get_format(self) -> List[str]:
        """Get the dataset format."""
        return self._dataset.visible_columns

    def set_format(self, columns: List[str]):
        """Set the dataset format."""
        # TODO(karan): change `cache`
        if "cache" in self.column_names:
            return self._dataset.set_format(columns=columns + ["cache"])
        return self._dataset.set_format(columns=columns)

    def reset_format(self):
        """Set the dataset format."""
        return self._dataset.reset_format()

    def set_visible_rows(self, indices: Sequence):
        """Set the visible rows in the dataset."""
        self._dataset.set_visible_rows(indices)

    def reset_visible_rows(self):
        """Reset to make all rows visible."""
        self._dataset.reset_visible_rows()

    def add_column(self, column: str, values: List):
        """Add a column to the dataset."""
        self._dataset.add_column(column, values)

    def _add_index(self):
        """Add an index to the dataset."""
        self.add_column("index", [str(i) for i in range(len(self))])

    def head(self, n: int, columns: List[str] = None):
        """View the first `n` examples of the dataset."""
        with self.format(columns):
            return pd.DataFrame(self[:n])

    @staticmethod
    def add_index(example, index):
        """Add an index to the dataset."""
        if "index" not in example:
            example["index"] = str(index)
        return example

    def _create_logdir(self):
        """Create and assign a directory for logging this dataset's files."""
        if self.identifier.name == "RGDataset":
            # TODO(karan): handle temporarily constructed datasets differently
            self.logdir /= str(self.identifier)
            self.logdir.mkdir(parents=True, exist_ok=True)
        else:
            self.logdir /= str(self.identifier)
            self.logdir.mkdir(parents=True, exist_ok=True)

    def _autobuild_identifier(self) -> Identifier:
        """Automatically build an identifier for the dataset using available
        information."""
        # Look for a name, otherwise assign a default
        _name = (
            self._dataset.info.builder_name
            if self._dataset.info.builder_name
            else "RGDataset"
        )

        # Check for split, version information
        split = str(self._dataset.split) if self._dataset.split else None
        version = self._dataset.version if self._dataset.version else None

        # Add all available information to kwargs dict
        kwargs = {}
        if split:
            kwargs["split"] = split
        if version:
            kwargs["version"] = version

        # Create identifier
        return Identifier(_name=_name, **kwargs)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]

    def __repr__(self):
        return (
            f"RG{self.__class__.__name__}["
            f"num_rows: {self.num_rows}]({self.identifier})"
        )

    @property
    def column_names(self):
        """Name of the columns in the dataset."""
        return self._dataset.column_names

    @property
    def has_index(self) -> bool:
        """Check if the dataset has an index column."""
        if self._dataset.column_names:
            return "index" in self._dataset.column_names
        # Just return True if the dataset is empty
        return True

    @classmethod
    def uncached_batch(cls, batch: Batch, copy=True) -> Batch:
        """Return batch with the "cache" and "slices" columns removed."""
        return tz.keyfilter(
            lambda k: k not in ["cache", "slices"], deepcopy(batch) if copy else batch
        )

    @classmethod
    def uncached_example(cls, example: Dict, copy=True) -> Dict:
        """Return example with the "cache" and "slices" columns removed."""
        return tz.keyfilter(
            lambda k: k not in ["cache", "slices"],
            deepcopy(example) if copy else example,
        )

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List datasets on Huggingface datasets.

        Returns: list of datasets
        """
        return datasets.list_datasets()

    @classmethod
    def load_dataset(
        cls,
        *args,
        dataset_fmt: str = "in_memory",
        **kwargs,
    ):
        """Create a Dataset using Huggingface datasets.load_dataset(..). Loads
        any dataset available in Huggingface Dataset Hub.

        Use this instead of datasets.load_dataset, so

        dict_of_datasets = datasets.load_dataset('boolq')

        becomes

        dict_of_datasets = Dataset.load_dataset('boolq')
        """
        # Load the dataset
        dataset = datasets.load_dataset(*args, **kwargs)

        if isinstance(dataset, dict):
            return dict(
                map(
                    lambda t: (t[0], cls(t[1], dataset_fmt=dataset_fmt)),
                    dataset.items(),
                )
            )
        else:
            return cls(dataset, dataset_fmt=dataset_fmt)

    @classmethod
    def from_datasets(
        cls,
        dataset: datasets.Dataset,
        identifier: Identifier = None,
        dataset_fmt: str = None,
    ) -> Dataset:
        """Create a Dataset from a Huggingface datasets.Dataset."""
        return cls(
            dataset,
            identifier=identifier,
            dataset_fmt=dataset_fmt,
        )

    @classmethod
    def from_jsonl(
        cls,
        json_path: str,
        identifier: Identifier = None,
        dataset_fmt: str = "in_memory",
    ) -> Dataset:
        """Load a dataset from a .jsonl file on disk, where each line of the
        json file consists of a single example."""

        if dataset_fmt == "in_memory":
            # Load the .jsonl file
            with open(json_path) as f:
                data = [json.loads(line) for line in f]

            return cls(
                data,
                identifier=identifier,
                dataset_fmt=dataset_fmt,
            )

        elif dataset_fmt == "datasets":
            # Use jsonarrow to directly load the json
            return cls(
                jsonarrow.read_json(json_path),
                identifier=identifier,
                dataset_fmt=dataset_fmt,
            )
        else:
            raise NotImplementedError

    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        identifier: Identifier = None,
        dataset_fmt: str = "in_memory",
    ) -> Dataset:
        """Convert a batch to a Dataset."""

        if dataset_fmt == "in_memory":
            return cls(batch, identifier=identifier, dataset_fmt=dataset_fmt)
        elif dataset_fmt == "datasets":
            return cls(table(batch), identifier=identifier, dataset_fmt=dataset_fmt)
        else:
            raise NotImplementedError

    @classmethod
    def from_batches(
        cls,
        batches: Sequence[Batch],
        identifier: Identifier = None,
        dataset_fmt: str = "in_memory",
    ) -> Dataset:
        """Convert a list of batches to a dataset."""

        return cls.from_batch(
            tz.merge_with(
                tz.compose(list, tz.concat),
                *batches,
            ),
            identifier=identifier,
            dataset_fmt=dataset_fmt,
        )

    @classmethod
    def from_dict(
        cls,
        d: Dict,
        identifier: Identifier = None,
        dataset_fmt: str = "in_memory",
    ) -> Dataset:
        """Convert a dictionary to a dataset.

        Alias for Dataset.from_batch(..).
        """
        return cls.from_batch(
            batch=d,
            identifier=identifier,
            dataset_fmt=dataset_fmt,
        )

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        identifier: Identifier = None,
        dataset_fmt: str = "in_memory",
    ):
        """Create a Dataset from a pandas DataFrame."""
        return cls.from_batch(
            df.to_dict("list"),
            identifier=identifier,
            dataset_fmt=dataset_fmt,
        )

    @classmethod
    def from_feather(
        cls,
        path: str,
        identifier: Identifier = None,
        dataset_fmt: str = "in_memory",
    ):
        """Create a Dataset from a feather file."""
        return cls.from_batch(
            pd.read_feather(path).to_dict("list"),
            identifier=Identifier("Feather", path=path)
            if not identifier
            else identifier,
            dataset_fmt=dataset_fmt,
        )

    def to_pandas(self) -> pd.DataFrame:
        """Convert a Dataset to a pandas DataFrame."""
        return pd.DataFrame(self[:])

    def batch(self, batch_size: int = 32):
        """Batch the dataset.

        Args:
            batch_size: integer batch size

        Returns:
        """
        for i in range(0, len(self), batch_size):
            yield self[i : i + batch_size]

    def update(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        # input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        # remove_columns: Optional[List[str]] = None,
        # keep_in_memory: bool = False,
        # load_from_cache_file: bool = True,
        # cache_file_name: Optional[str] = None,
        # writer_batch_size: Optional[int] = 1000,
        # features: Optional[Features] = None,
        # disable_nullable: bool = False,
        # fn_kwargs: Optional[dict] = None,
        # num_proc: Optional[int] = None,
        # suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        # new_fingerprint: Optional[str] = None,
        # **kwargs,
    ) -> Dataset:
        """Map on the dataset."""

        assert isinstance(
            self._dataset, InMemoryDataset
        ), "Cannot apply .update() if format isn't InMemoryDataset."
        # Compute the map using the underlying dataset's .map()
        output = self._dataset.update(
            function=function,
            with_indices=with_indices,
            # input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            # remove_columns=remove_columns,
            # keep_in_memory=keep_in_memory,
            # load_from_cache_file=load_from_cache_file,
            # cache_file_name=cache_file_name,
            # writer_batch_size=writer_batch_size,
            # features=features,
            # disable_nullable=disable_nullable,
            # fn_kwargs=fn_kwargs,
            # num_proc=num_proc,
            # suffix_template=suffix_template,
            # new_fingerprint=new_fingerprint,
        )

        if isinstance(output, datasets.Dataset):
            dataset = copy(self)
            dataset._dataset = output
        elif isinstance(output, InMemoryDataset):
            dataset = copy(self)
            dataset._dataset = output
        elif output is None:
            dataset = self
        else:
            raise NotImplementedError("Unrecognized dataset.")

        return dataset

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict, List]:
        """Map on the dataset."""

        assert isinstance(
            self._dataset, InMemoryDataset
        ), "Cannot apply .update() if format isn't InMemoryDataset."
        # Compute the map using the underlying dataset's .map()
        return self._dataset.map(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            suffix_template=suffix_template,
            new_fingerprint=new_fingerprint,
        )

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batch_size: Optional[int] = 1000,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
    ) -> Dataset:
        """Filter operation on the dataset."""
        # Compute the filter using the underlying dataset's .filter()
        output = self._dataset.filter(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batch_size=batch_size,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            suffix_template=suffix_template,
            new_fingerprint=new_fingerprint,
        )

        if isinstance(output, datasets.Dataset):
            dataset = copy(self)
            dataset._dataset = output
        elif isinstance(output, InMemoryDataset):
            dataset = copy(self)
            dataset._dataset = output
        else:
            raise NotImplementedError("Unrecognized dataset generated after .filter().")

        return dataset

    @classmethod
    def interleave(
        cls,
        datasets: List[Dataset],
        identifier: Identifier,
    ) -> Dataset:

        """Interleave a list of datasets."""
        return cls.from_batch(
            tz.merge_with(
                tz.compose(list, tz.interleave),
                *[dataset[:] for dataset in datasets],
            ),
            identifier=identifier,
        )

    @classmethod
    def chain(
        cls,
        datasets: List[Dataset],
        identifier: Identifier,
    ) -> Dataset:

        """Chain a list of datasets."""
        return cls.from_batch(
            tz.merge_with(
                tz.compose(list, tz.concat),
                *[dataset[:] for dataset in datasets],
            ),
            identifier=identifier,
        )

    @classmethod
    def load_from_disk(cls, path: str = None, identifier: Identifier = None) -> Dataset:
        """Load a dataset stored on disk."""
        assert (
            path or identifier and not (path and identifier)
        ), "Pass one of `path` or `identifier`."

        if identifier:
            # Use the default logdir to create a path to the dataset
            path = cls.logdir / str(identifier)
            if not os.path.exists(str(path)):
                raise OSError(f"Path {path} does not exist.")

        # Create an empty state
        state = {}

        # Load the metadata
        metadata = json.load(open(os.path.join(path, "metadata.json")))

        # Load the data
        if metadata["_dataset_fmt"] == "in_memory":
            state["_dataset"] = InMemoryDataset.load_from_disk(
                os.path.join(path, "_dataset")
            )
        elif metadata["_dataset_fmt"] == "datasets":
            state["_dataset"] = datasets.Dataset.load_from_disk(
                os.path.join(path, "_dataset")
            )
        else:
            raise NotImplementedError(
                f"`dataset_fmt` {metadata['_dataset_fmt']} not recognized."
            )

        # Merge the metadata with the state
        state = {**state, **metadata}

        # Create an empty dataset
        dataset = cls()
        dataset.__setstate__(state)

        return dataset

    def save_to_disk(self, path: str = None) -> None:
        """Save a dataset to disk."""
        if path is None:
            path = str(self.logdir)

        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Get the dataset state
        state = self.__getstate__()

        # Save the data to disk
        os.makedirs(os.path.join(path, "_dataset"), exist_ok=True)
        state["_dataset"].save_to_disk(os.path.join(path, "_dataset"))

        # Save the metadata to disk
        json.dump(
            {k: v for k, v in state.items() if k != "_dataset"},
            open(os.path.join(path, "metadata.json"), "w"),
        )

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {
            "interactions",
            "_identifier",
            "_dataset",
            "_dataset_fmt",
        }

    @classmethod
    def _assert_state_keys(cls, state: Dict) -> None:
        """Assert that a state contains all required keys."""
        assert (
            set(state.keys()) == cls._state_keys()
        ), f"State must contain all state keys: {cls._state_keys()}."

    def __getstate__(self):
        """Get the current state of the dataset."""

        state = {
            # Interaction state
            "interactions": self.dumps_interactions(),
            # Identifier
            "_identifier": self.identifier.dumps() if self.identifier else None,
            # Dataset
            "_dataset": self._dataset,
            # Dataset format
            "_dataset_fmt": self._dataset_fmt,
        }
        Dataset._assert_state_keys(state)

        return state

    def __setstate__(self, state):
        """Set the current state of the dataset."""
        # Check that the state contains all keys
        Dataset._assert_state_keys(state)

        # Load the interactions
        self.interactions = self.loads_interactions(state["interactions"]).interactions

        # Load the identifier
        self._identifier = (
            Identifier.loads(state["_identifier"]) if state["_identifier"] else None
        )

        # Load the dataset
        self._dataset = state["_dataset"]

        # Set the dataset format
        self._dataset_fmt = state["_dataset_fmt"]

        # Update the logging directory
        self.logdir = Dataset.logdir / str(self.identifier)


def transpose_batch(batch: Batch):
    """Transpose a batch of data from a dict of lists to a list of dicts.

    Args:
        batch: batch of data which is a dictionary mapping columns to lists

    Returns: list of dicts, each dict corresponding to a single example
    """
    return [dict(zip(batch, t)) for t in zip(*batch.values())]
