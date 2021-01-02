from __future__ import annotations

import json
import os
import pathlib
import pickle
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Sequence, Union

import cytoolz as tz
import datasets
from datasets import Features
from datasets.arrow_writer import ArrowWriter
from pyarrow import json as jsonarrow
from pyarrow import table
from tqdm import tqdm

from robustnessgym.core.constants import (
    ATTACK,
    CACHEDOPS,
    SLICEBUILDERS,
    SUBPOPULATION,
    TRANSFORMATION,
)
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import persistent_hash, strings_as_json


class InteractionTape:
    def __init__(self):
        # Keep track of the history
        self.history = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(interactions={len(self.history)})"

    def __hash__(self):
        val = 0
        for (identifier, json_columns) in self.history:
            val ^= persistent_hash(str(identifier) + str(json_columns))
        return val

    def dumps(self):
        return json.dumps(
            {
                json.dumps((identifier.dumps(), json_columns)): idx
                for (identifier, json_columns), idx in self.history.items()
            }
        )

    @classmethod
    def loads(cls, s: str):
        tape = InteractionTape()
        history = json.loads(s)
        history = {
            tuple(json.loads(json_tuple)): idx for json_tuple, idx in history.items()
        }
        tape.history = {
            (Identifier.loads(identifier), json_columns): idx
            for (identifier, json_columns), idx in history.items()
        }

        return tape

    def update(self, identifier: Union[str, Identifier], columns: List[str]) -> None:
        """Update the interaction tape with information about an interaction.

        Args:
            identifier: Identifier for the interaction used.
            columns: list of columns on which the interaction was applied.

        Returns: True if the interaction was added to the tape, False if it was
        already applied before.
        """
        if isinstance(identifier, str):
            identifier = Identifier(_name=identifier)
        elif isinstance(identifier, Identifier):
            pass
        else:
            raise ValueError(
                f"Parameter `identifier` should be an instance of class Identifier "
                f"or str, "
                f"not {type(identifier)}."
            )

        # Dump the column names to JSON
        json_columns = strings_as_json(strings=columns)

        # Check if the entry is not in the history
        if (identifier, json_columns) not in self.history:
            # Give it the next index
            self.history[(identifier, json_columns)] = len(self.history)

    def check(self, identifier: Union[str, Identifier], columns: List[str]) -> bool:
        """

        Args:
            identifier:
            columns:

        Returns:

        """
        if not (isinstance(identifier, str) or isinstance(identifier, Identifier)):
            raise ValueError(
                f"Parameter `identifier` should be an instance of class Identifier "
                f"or str, "
                f"not {type(identifier)}."
            )

        # Dump the column names to JSON
        json_columns = strings_as_json(strings=columns)

        # Check if the entry is already in the history
        if (identifier, json_columns) in self.history:
            return True
        return False


class InteractionTapeHierarchyMixin:
    def __init__(self):
        self.interactions = {
            CACHEDOPS: InteractionTape(),
            SLICEBUILDERS: {
                SUBPOPULATION: InteractionTape(),
                TRANSFORMATION: InteractionTape(),
                ATTACK: InteractionTape(),
            },
        }

    def hash_interactions(self):
        v = 0
        for path in [
            [CACHEDOPS],
            [SLICEBUILDERS, SUBPOPULATION],
            [SLICEBUILDERS, TRANSFORMATION],
            [SLICEBUILDERS, ATTACK],
        ]:
            v ^= self.fetch_tape(path=path).__hash__()
        return v

    def dumps_interactions(self):
        return json.dumps(
            {
                CACHEDOPS: self.interactions[CACHEDOPS].dumps(),
                SLICEBUILDERS: {
                    SUBPOPULATION: self.interactions[SLICEBUILDERS][
                        SUBPOPULATION
                    ].dumps(),
                    TRANSFORMATION: self.interactions[SLICEBUILDERS][
                        TRANSFORMATION
                    ].dumps(),
                    ATTACK: self.interactions[SLICEBUILDERS][ATTACK].dumps(),
                },
            }
        )

    @classmethod
    def loads_interactions(cls, s: str) -> InteractionTapeHierarchyMixin:
        tape_hierarchy = InteractionTapeHierarchyMixin()
        interactions = json.loads(s)
        tape_hierarchy.interactions = {
            CACHEDOPS: InteractionTape.loads(interactions[CACHEDOPS]),
            SLICEBUILDERS: {
                SUBPOPULATION: InteractionTape.loads(
                    interactions[SLICEBUILDERS][SUBPOPULATION]
                ),
                TRANSFORMATION: InteractionTape.loads(
                    interactions[SLICEBUILDERS][TRANSFORMATION]
                ),
                ATTACK: InteractionTape.loads(interactions[SLICEBUILDERS][ATTACK]),
            },
        }
        return tape_hierarchy

    def update_tape(
        self,
        path: List[str],
        identifiers: Union[Identifier, List[Identifier]],
        columns: List[str],
    ):
        """Update the tape.

        Args:
            path: Location of the InteractionTape in the hierarchy.
            identifiers:
            columns:

        Returns:
        """
        # Fetch the tape
        tape = self.fetch_tape(path=path)

        # Update it
        if isinstance(identifiers, Identifier) or isinstance(identifiers, str):
            return tape.update(identifier=identifiers, columns=columns)
        else:
            return [
                tape.update(identifier=identifier, columns=columns)
                for identifier in identifiers
            ]

    def check_tape(
        self,
        path: List[str],
        identifiers: Union[Identifier, List[Identifier]],
        columns: List[str],
    ):
        """Check the tape.

        Args:

            path:
            identifiers:
            columns:

        Returns:
        """
        # Fetch the tape
        tape = self.fetch_tape(path=path)

        # Check it
        if isinstance(identifiers, Identifier) or isinstance(identifiers, str):
            return tape.check(identifier=identifiers, columns=columns)
        else:
            return [
                tape.check(identifier=identifier, columns=columns)
                for identifier in identifiers
            ]

    def fetch_tape(self, path: List[str]) -> InteractionTape:
        """Fetch an InteractionTape.

        Args:
            path:

        Returns:
        """
        return tz.get_in(path, self.interactions)


Batch = Dict[str, List]
BatchOrDataset = Union[Batch, "Dataset"]


class Dataset(datasets.Dataset, InteractionTapeHierarchyMixin):
    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / "robustnessgym/datasets/"

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    def __init__(self, *args, identifier: Identifier = None, **kwargs):

        if len(args) == 1 and isinstance(args[0], datasets.Dataset):
            # Create a Dataset directly from an datasets.Dataset object
            self.__dict__ = args[0].__dict__.copy()
        else:
            super(Dataset, self).__init__(*args, **kwargs)

        # Call the superclass constructor
        InteractionTapeHierarchyMixin.__init__(self)

        self.identifier = (
            Identifier(
                _name=self.info.builder_name,
                split=str(self.split),
                version=self.version,
            )
            if not identifier
            else identifier
        )

        # Keep track of the original dataset keys
        self.original_columns = list(self.features.keys())

        # Add an index to the dataset
        dataset = self.map(self.add_index, with_indices=True)
        self.__dict__.update(dataset.__dict__)

        # TODO(karan): fix the identifier settings for Dataset
        if self.identifier is not None and not str(self.identifier).startswith("None"):
            self.logdir /= str(self.identifier)
            self.logdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def add_index(example, index):
        if "index" not in example:
            example["index"] = str(index)
        return example

    def __repr__(self):
        return (
            f"RobustnessGym{self.__class__.__name__}(num_rows: {self.num_rows}, "
            f"interactions: {self.interactions})"
        )

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
    def from_huggingface(cls, dataset: datasets.Dataset):
        """Create a Dataset from a Huggingface datasets.Dataset."""
        return cls(dataset.info.builder_name, dataset)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List datasets on Huggingface.

        Returns: list of datasets
        """
        return datasets.list_datasets()

    @classmethod
    def load_dataset(cls, *args, **kwargs):
        """Create a Dataset from any Huggingface nlp dataset source.

        Use this instead of datasets.load_dataset, so that

        dict_of_datasets = datasets.load_dataset('boolq')

        becomes

        dict_of_datasets = Dataset.load_dataset('boolq')
        """
        # Load the dataset
        dataset = datasets.load_dataset(*args, **kwargs)

        if isinstance(dataset, dict):
            return dict(
                map(
                    lambda t: (
                        t[0],
                        cls(
                            t[1],
                            identifier=Identifier(
                                _name=t[1].info.builder_name,
                                split=str(t[1].split),
                                version=t[1].version,
                            ),
                        ),
                    ),
                    dataset.items(),
                )
            )
        else:
            return cls(
                dataset,
                identifier=Identifier(
                    _name=dataset.info.builder_name,
                    split=str(dataset.split),
                    version=dataset.version,
                ),
            )

    @classmethod
    def from_json(cls, json_path: str, identifier: Identifier) -> Dataset:
        """Load a dataset from a JSON file on disk, where each line of the json
        file consists of a single example."""
        return cls(
            jsonarrow.read_json(json_path),
            identifier=identifier,
        )

    @classmethod
    def from_slice(cls):
        pass

    @classmethod
    def from_batch(cls, batch: Batch, identifier: Identifier) -> Dataset:
        """Convert a batch to a Dataset.

        TODO(karan): disable preprocessing in this case
        """
        return cls(table(batch), identifier=identifier)

    @classmethod
    def from_batches(
        cls, batches: Sequence[Batch], identifier: Identifier = None
    ) -> Dataset:
        """Convert a list of batches to a dataset."""
        return cls.from_batch(
            tz.merge_with(tz.concat, *batches),
            identifier=identifier,
        )

    def batch(self, batch_size: int = 32):
        """Batch the dataset.

        Args:
            batch_size: integer batch size

        Returns:
        """
        for i in range(0, len(self), batch_size):
            yield self[i : i + batch_size]

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
    ) -> Dataset:
        """Wrap map."""

        # Compute the map using datasets.Dataset's .map()
        output = datasets.Dataset.map(
            self,
            function,
            with_indices,
            input_columns,
            batched,
            batch_size,
            drop_last_batch,
            remove_columns,
            keep_in_memory,
            load_from_cache_file,
            cache_file_name,
            writer_batch_size,
            features,
            disable_nullable,
            fn_kwargs,
            num_proc,
            suffix_template,
            new_fingerprint,
        )

        if isinstance(output, datasets.Dataset):
            dataset = deepcopy(self)
            dataset.__dict__ = tz.merge(dataset.__dict__, output.__dict__)
            return dataset
        else:
            return output

    @classmethod
    def load(cls, path: str) -> Optional[Dataset]:
        try:
            with open(os.path.join(path, "split.p"), "rb") as f:
                return cls.from_file(
                    filename=os.path.join(path, "data.arrow"),
                    info=datasets.DatasetInfo.from_directory(path),
                    split=pickle.load(f),
                )
        except:  # noqa
            return None

    # def save_to_disk(self, dataset_path: str):
    #     return super(Dataset, self).save_to_disk(dataset_path)

    def save(self, path: str) -> None:
        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Taken from Huggingface datasets.Dataset
        # Prepare output buffer and batched writer in memory or on file if we update
        # the table
        writer = ArrowWriter(
            features=self.features,
            path=os.path.join(path, "data.arrow"),
            writer_batch_size=1000,
        )

        # Loop over single examples or batches and write to buffer/file if examples
        # are to be updated
        for i, example in tqdm(enumerate(self)):
            writer.write(example)

        writer.finalize()

        # Write DatasetInfo
        self.info.write_to_directory(path)

        # Write split to file
        with open(os.path.join(path, "split.p"), "wb") as f:
            pickle.dump(self.split, f)

    @classmethod
    def from_tfds(cls):
        # TODO(karan): v1 of robustness gym. Use it for image-based tasks, like clevr.
        pass

    @classmethod
    def interleave(cls, datasets: List[Dataset], identifier: Identifier) -> Dataset:
        """Interleave a list of datasets."""
        return cls.from_batch(
            tz.merge_with(tz.interleave, *[dataset[:] for dataset in datasets]),
            identifier=identifier,
        )

    @classmethod
    def chain(cls, datasets: List[Dataset], identifier: Identifier) -> Dataset:
        """Chain a list of datasets."""
        return cls.from_batch(
            tz.merge_with(tz.concat, *[dataset[:] for dataset in datasets]),
            identifier=identifier,
        )

    def __getstate__(self):
        state = super(Dataset, self).__getstate__()
        if "interactions" in state and not isinstance(state["interactions"], str):
            state["interactions"] = self.dumps_interactions()
        if "identifier" in state and isinstance(state["identifier"], Identifier):
            state["identifier"] = state["identifier"].dumps()
        if "_identifier" in state and isinstance(state["_identifier"], Identifier):
            state["_identifier"] = state["_identifier"].dumps()
        if "lineage" in state:
            state["lineage"] = [
                tuple(t[:1]) + (t[1].dumps(),) + (tuple(t[2:]) if len(t) > 2 else ())
                for t in state["lineage"]
            ]
        if "logdir" in state:
            state["logdir"] = ""
        return state

    def __setstate__(self, state):
        state = dict(state)
        if "interactions" in state and isinstance(state["interactions"], str):
            state["interactions"] = self.loads_interactions(
                state["interactions"]
            ).interactions
        if "identifier" in state and isinstance(state["identifier"], str):
            state["identifier"] = Identifier.loads(state["identifier"])
        if "_identifier" in state:
            try:
                state["_identifier"] = Identifier.loads(state["_identifier"])
            except:  # noqa
                pass
        if "lineage" in state:
            try:
                state["lineage"] = [
                    tuple(t[:1])
                    + (Identifier.loads(t[1]),)
                    + (tuple(t[2:]) if len(t) > 2 else ())
                    for t in state["lineage"]
                ]
            except:  # noqa
                pass
        if "logdir" in state:
            try:
                state["logdir"] = (
                    pathlib.Path.home()
                    / f"robustnessgym/datasets/{str(state['identifier'])}"
                )
            except:  # noqa
                state["logdir"] = (
                    pathlib.Path.home()
                    / f"robustnessgym/datasets/{str(state['_identifier'])}"
                )
        super(Dataset, self).__setstate__(state)

    @classmethod
    def load_from_disk(cls, dataset_path: str) -> Dataset:
        """Load the dataset from a dataset directory.

        Args:
            dataset_path (``str``): path of the dataset directory where the dataset
            will be loaded from
        """
        with open(os.path.join(dataset_path, "state.json"), "r") as state_file:
            state = json.load(state_file)
        with open(
            os.path.join(dataset_path, "dataset_info.json"), "r"
        ) as dataset_info_file:
            dataset_info = json.load(dataset_info_file)
        state["_info"] = json.dumps(dataset_info)
        dataset = cls.from_dict({})
        state = {
            k: state[k] for k in dataset.__dict__.keys()
        }  # in case we add new fields
        # Change path to absolute path
        for data_file in state.get("_data_files", []) + state.get(
            "_indices_data_files", []
        ):
            data_file["filename"] = os.path.join(dataset_path, data_file["filename"])
        dataset.__setstate__(state)
        dataset.logdir = (
            pathlib.Path.home() / f"robustnessgym/datasets/{str(dataset.identifier)}"
        )
        return dataset


def transpose_batch(batch: Batch):
    """Transpose a batch of data from a dict of lists to a list of dicts.

    Args:
        batch: batch of data which is a dictionary mapping columns to lists

    Returns: list of dicts, each dict corresponding to a single example
    """
    return [dict(zip(batch, t)) for t in zip(*batch.values())]
