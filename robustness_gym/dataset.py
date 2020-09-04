from __future__ import annotations

import os
import pickle
from copy import deepcopy
from typing import *

import cytoolz as tz
import nlp
import pyarrow as pa
from nlp.arrow_writer import ArrowWriter
from pyarrow import json as jsonarrow, table
from tqdm import tqdm

from robustness_gym.constants import *
from robustness_gym.identifier import Identifier
from robustness_gym.tools import strings_as_json


class InteractionTape:

    def __init__(self):
        # Keep track of the history
        self.history = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(interactions={len(self.history)})"

    def update(self,
               identifier: Identifier,
               keys: List[str]):

        # Dump the keys to JSON
        json_keys = strings_as_json(strings=keys)

        # Check if the entry is not in the history
        if (identifier, json_keys) not in self.history:
            # Give it the next index
            self.history[(identifier, json_keys)] = len(self.history)
            return True
        return False

    def check(self,
              identifier: Identifier,
              keys: List[str]):

        # Dump the keys to JSON
        json_keys = strings_as_json(strings=keys)

        # Check if the entry is already in the history
        if (identifier, json_keys) in self.history:
            return 0
        return 1


class InteractionTapeHierarchyMixin:

    def __init__(self):
        self.interactions = {
            CACHED_OPS: InteractionTape(),
            SLICEMAKERS: {
                SUBPOPULATION: InteractionTape(),
                AUGMENTATION: InteractionTape(),
                ATTACK: InteractionTape(),
            }
        }

    def update_tape(
            self,
            path: List[str],
            identifier: Identifier,
            keys: List[str],
    ):
        """
        Update the tape.

        Args:
            path: Location of the InteractionTape in the hierarchy.
            identifier:
            keys:

        Returns:

        """
        return self.fetch_tape(path=path).update(identifier=identifier, keys=keys)

    def check_tape(
            self,
            path: List[str],
            identifier: Identifier,
            keys: List[str]
    ):
        """
        Check the tape.

        Args:
            path:
            identifier:
            keys:

        Returns:

        """
        return self.fetch_tape(path=path).check(identifier=identifier, keys=keys)

    def fetch_tape(
            self,
            path: List[str]
    ):
        """
        Fetch an InteractionTape.

        Args:
            path:

        Returns:

        """
        return tz.get_in(path, self.interactions)


class Dataset(nlp.Dataset, InteractionTapeHierarchyMixin):

    def __init__(self, *args, **kwargs):

        if len(args) == 1 and isinstance(args[0], nlp.Dataset):
            # Create a Dataset directly from an nlp.Dataset object
            self.__dict__ = args[0].__dict__.copy()
        else:
            super(Dataset, self).__init__(*args, **kwargs)

        # Call the superclass constructor
        InteractionTapeHierarchyMixin.__init__(self)

        # Keep track of the original dataset keys
        self.original_keys = list(self.features.keys())
        self.num_slices = 0

        # Add an index to the dataset
        dataset = self.map(self.add_index, with_indices=True)
        self.__dict__.update(dataset.__dict__)

    @staticmethod
    def add_index(example, index):
        if 'index' not in example:
            example['index'] = str(index)
        return example

    def __repr__(self):
        return f"{self.__class__.__name__}(features: {self.features}, " \
               f"num_rows: {self.num_rows}, " \
               f"num_slices: {self.num_slices})"

    @classmethod
    def uncached_batch(cls,
                       batch: Dict[str, List],
                       copy=True) -> Dict[str, List]:
        """
        Return batch with the "cache" and "slices" keys removed.
        """
        return tz.keyfilter(lambda k: k not in ['cache', 'slices'], deepcopy(batch) if copy else batch)

    @classmethod
    def uncached_example(cls,
                         example: Dict,
                         copy=True) -> Dict:
        """
        Return example with the "cache" and "slices" keys removed.
        """
        return tz.keyfilter(lambda k: k not in ['cache', 'slices'], deepcopy(example) if copy else example)

    @classmethod
    def from_nlp(cls,
                 dataset: nlp.Dataset):
        """
        Create a Dataset from a Huggingface nlp.Dataset.
        """
        return cls(dataset)

    @classmethod
    def load_dataset(cls,
                     *args,
                     **kwargs):
        """
        Create a Dataset from any Huggingface nlp dataset source.

        Use this instead of nlp.load_dataset, so that

        dict_of_datasets = nlp.load_dataset('boolq')

        becomes

        dict_of_datasets = Dataset.load_dataset('boolq')
        """
        # Load the dataset
        dataset = nlp.load_dataset(*args, **kwargs)

        if isinstance(dataset, dict):
            return dict(map(lambda t: (t[0], cls(t[1])), dataset.items()))
        else:
            return cls(dataset)

    @classmethod
    def from_json(cls,
                  json_path: str) -> Dataset:
        """
        Load a dataset from a JSON file on disk, where each line of the json file consists of a single example.
        """
        return cls(jsonarrow.read_json(json_path))

    @classmethod
    def from_slice(cls):
        pass

    @classmethod
    def from_batch(cls,
                   batch: Dict[str, List]) -> Dataset:
        """
        Convert a batch to a Dataset.

        TODO(karan): disable preprocessing in this case
        """
        return cls(table(batch))

    @classmethod
    def from_batches(cls,
                     batches: Sequence[Dict[str, List]]) -> Dataset:
        """
        Convert a list of batches to a dataset.
        """
        return cls.from_batch(tz.merge_with(tz.concat, *batches))

    def map(self,
            function,
            with_indices: bool = False,
            batched: bool = False,
            batch_size: Optional[int] = 1000,
            remove_columns: Optional[List[str]] = None,
            keep_in_memory: bool = False,
            load_from_cache_file: bool = True,
            cache_file_name: Optional[str] = None,
            writer_batch_size: Optional[int] = 1000,
            arrow_schema: Optional[pa.Schema] = None,
            disable_nullable: bool = True,
            **kwargs
            ) -> Dataset:
        """
        Wrap map.
        """

        # Compute the map using nlp Dataset's .map()
        output = nlp.Dataset.map(
            self,
            function,
            with_indices,
            batched,
            batch_size,
            remove_columns,
            keep_in_memory,
            load_from_cache_file,
            cache_file_name,
            writer_batch_size,
            arrow_schema,
            disable_nullable,
        )

        if isinstance(output, nlp.Dataset):
            dataset = deepcopy(self)
            dataset.__dict__ = tz.merge(dataset.__dict__, output.__dict__)
            return dataset
        else:
            return output

    @classmethod
    def load(cls, path: str) -> Optional[Dataset]:
        try:
            return cls.from_file(filename=os.path.join(path, 'data.arrow'),
                                 info=nlp.DatasetInfo.from_directory(path),
                                 split=pickle.load(open(os.path.join(path, 'split.p'), 'rb')))
        except:
            return None

    def save(self, path: str) -> None:
        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Taken from Huggingface nlp.Dataset
        # Prepare output buffer and batched writer in memory or on file if we update the table
        writer = ArrowWriter(features=self.features, path=os.path.join(
            path, 'data.arrow'), writer_batch_size=1000)

        # Loop over single examples or batches and write to buffer/file if examples are to be updated
        for i, example in tqdm(enumerate(self)):
            writer.write(example)

        writer.finalize()

        # Write DatasetInfo
        self.info.write_to_directory(path)

        # Write split to file
        pickle.dump(self.split, open(os.path.join(path, 'split.p'), 'wb'))

    @classmethod
    def from_tfds(cls):
        # TODO(karan): v1 of robustness gym. Use it for image-based tasks, like clevr.
        pass

    @classmethod
    def interleave(cls, datasets: List[Dataset]) -> Dataset:
        """
        Interleave a list of datasets.
        """
        return cls.from_batch(
            tz.merge_with(tz.interleave,
                          *[dataset[:] for dataset in datasets])
        )

    @classmethod
    def chain(cls, datasets: List[Dataset]) -> Dataset:
        """
        Chain a list of datasets.
        """
        return cls.from_batch(
            tz.merge_with(tz.concat,
                          *[dataset[:] for dataset in datasets])
        )
