from __future__ import annotations

import os
import pickle
from copy import deepcopy
from functools import partial
from typing import *

import cytoolz as tz
import nlp
import pyarrow as pa
import spacy
import torch
import json
import hashlib
from allennlp.predictors import Predictor
from nlp.arrow_writer import ArrowWriter
from pyarrow import json as jsonarrow, table
from quinine.common.utils import rmerge
from tqdm import tqdm


def persistent_hash(s: str):
    """
    Compute a hash that persists across multiple Python sessions for a string.
    """
    return int(hashlib.sha224(s.encode()).hexdigest(), 16)


class CachedOperation:
    identifier: str

    def __init__(self,
                 identifier,
                 apply_fn=None):
        # Set the identifier for the preprocessor
        self.identifier = identifier

        if apply_fn:
            self.apply = apply_fn

    @staticmethod
    def update_cache(batch: Dict[str], updates: List[Dict]) -> Dict[str]:
        """
        Updates the cache of preprocessed information stored with each example in a batch.

        - batch must contain a key called 'cache' that maps to a dictionary.
        - batch['cache'] is a list of dictionaries, one per example
        - updates is a list of dictionaries, one per example
        """
        if 'cache' not in batch:
            batch['cache'] = [{} for _ in range(len(batch['index']))]

        assert 'cache' in batch, "Examples must have a key called 'cache'."
        # assert len(batch['cache']) == len(updates), "Number of examples must equal the number of updates."

        # For each example, recursively merge the example's original cache dictionary with the update dictionary
        batch['cache'] = [rmerge(cache_dict, update_dict)
                          for cache_dict, update_dict in zip(batch['cache'], updates)]

        return batch

    def __hash__(self):
        """
        Compute a hash value for the cached operation object.
        """
        return persistent_hash(self.identifier)

    def get_cache_hash(self, keys: Optional[List[str]] = None):
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
                        batch_size: int = 32):
        """
        Apply the cached operation to a dataset.
        """

        # Apply the cached operation
        # Pass a file name for the cache: the automatically generated cache filename changes across sessions,
        # since the hash value of a class method is not fixed.
        return dataset.map(partial(self.process_batch, keys=keys), batched=True, batch_size=batch_size,
                           cache_file_name=self.get_cache_file_name(keys=keys))

    def process_batch(self,
                      batch: Dict[str, List],
                      keys: List[str]) -> Dict[str, List]:
        """
        Apply the cached operation to a batch.
        """
        assert len(set(keys) - set(batch.keys())
                   ) == 0, "Any key in 'keys' must be present in 'batch'."

        # Run the cached operation and get outputs
        processed_outputs = self.apply(*[batch[key] for key in keys])

        # Construct updates
        updates = [{self.identifier: {json.dumps(keys) if len(keys) > 1 else keys[0]: val}}
                   for val in processed_outputs]

        # Update the cache and return the updated batch
        return self.update_cache(batch=batch, updates=updates)

    def apply(self, *args, **kwargs) -> List:
        """
        Implements the core functionality of the cached operation.
        """
        pass

    def __call__(self, batch_or_dataset, keys):

        if isinstance(batch_or_dataset, Dataset):
            return self.process_dataset(dataset=batch_or_dataset,
                                        keys=keys)
        elif isinstance(batch_or_dataset, Dict):
            return self.process_batch(batch=batch_or_dataset,
                                      keys=keys)
        else:
            raise NotImplementedError


class Spacy(CachedOperation):

    def __init__(self,
                 name: str = "en_core_web_sm"):
        super(Spacy, self).__init__(identifier='Spacy')

        # Load up the spacy module
        spacy.prefer_gpu()
        self.name = name
        self.nlp = spacy.load(name)

    def __hash__(self):
        """
        For Spacy, the hash value includes the name of the module being loaded.
        """
        val = super(Spacy, self).__hash__()
        return val ^ persistent_hash(self.name)

    def apply(self, text_batch) -> List:
        # Apply spacy's pipe method to process the examples
        docs = list(self.nlp.pipe(text_batch))

        # Convert the docs to json
        return [val.to_json() for val in docs]


class AllenConstituencyParser(CachedOperation):

    def __init__(self):
        super(AllenConstituencyParser, self).__init__(
            identifier='AllenConstituencyParser')

        # Set up AllenNLP's constituency parser
        if torch.cuda.is_available():
            self.constituency_parser = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
                cuda_device=0
            )
        else:
            self.constituency_parser = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
            )

    def apply(self, text_batch) -> List:
        # Apply the constituency parser
        parse_trees = self.constituency_parser.predict_batch_json([{'sentence': text}
                                                                   for text in text_batch])

        # Extract the tree from the output of the constituency parser
        return [val['trees'] for val in parse_trees]


class StripText(CachedOperation):

    def __init__(self):
        super(StripText, self).__init__(identifier='StripText')

    def apply(self, text_batch) -> List:
        # Clean up each text with a simple function
        stripped = list(
            map(
                lambda text: text.lower().replace(".", "").replace(
                    "?", "").replace("!", "").replace(",", ""),
                text_batch
            )
        )

        # Return the stripped sentences
        return stripped


class TextBlob(CachedOperation):

    def __init__(self):
        super(TextBlob, self).__init__(identifier='TextBlob')

    def apply(self, text_batch) -> List:
        pass


# class PreprocessingMixin:
#     preprocessors = ['spacy', 'textblob', 'striptext', 'constituency_parser']

#     def preprocess(self, examples: Dict[str], keys: List[str]):
#         for key in keys:
#             examples = self.strip_text(examples, key)
#             examples = self.spacy_pipe(examples, key)
#             # examples = self.constituency_parse(examples, key)

#         return examples


class DatasetHelpersMixin:

    @staticmethod
    def add_slices_key(example):
        example['slices'] = {}
        return example

    @staticmethod
    def add_cache_key(example):
        example['cache'] = {}
        return example

    @staticmethod
    def add_spacy_key(example):
        example['cache']['spacy'] = {}
        return example

    @staticmethod
    def add_index(example, index):
        if 'index' not in example:
            example['index'] = str(index)
        return example


class Dataset(nlp.Dataset,
              DatasetHelpersMixin,
              #   PreprocessingMixin
              ):

    def __init__(self, *args, **kwargs):

        if len(args) == 1 and isinstance(args[0], nlp.Dataset):
            # Create a Dataset directly from an nlp.Dataset object
            self.__dict__ = args[0].__dict__.copy()
        else:
            super(Dataset, self).__init__(*args, **kwargs)

        # Keep track of the original dataset keys
        self.original_keys = self.schema.names
        self.num_slices = 0

        # Keep track of slicers that were executed on the dataset
        self.history = {
            'ops': {},
            'slicers': {
                'filters': {},
                'augmentations': {},
                'attacks': {},
            }
        }

        # Add an index to the dataset
        dataset = self.map(DatasetHelpersMixin.add_index, with_indices=True)
        self.__dict__.update(dataset.__dict__)

    @staticmethod
    def add_index(example, index):
        if 'index' not in example:
            example['index'] = str(index)
        return example

    def __repr__(self):
        schema_str = dict((a, str(b)) for a, b in zip(
            self._data.schema.names, self._data.schema.types))
        return f"{self.__class__.__name__}(schema: {schema_str}, num_rows: {self.num_rows}, num_slices: {self.num_slices})"

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
            ):
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
            self.__dict__ = tz.merge(self.__dict__, output.__dict__)
            return self
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
        writer = ArrowWriter(schema=self.schema, path=os.path.join(
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

    # def initialize(self, keys):
    #
    #     PreprocessingMixin.__init__(self)
    #
    #     # For convenience
    #     dataset = self
    #
    #     # Add an index to the dataset
    #     dataset = dataset.map(DatasetHelpersMixin.add_index, with_indices=True)
    #
    #     # Add a dictionary for keeping track of slice membership
    #     dataset = dataset.map(DatasetHelpersMixin.add_slices_key)
    #
    #     # Add a dictionary for keeping track of cached information
    #     dataset = dataset.map(DatasetHelpersMixin.add_cache_key)
    #
    #     # Apply an expensive preprocessing step using Spacy
    #     dataset = dataset.map(partial(self.preprocess, keys=keys), batched=True, batch_size=32)
    #
    #     self.__dict__.update(dataset.__dict__)

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

    def update_history(self, slicer, keys):
        """
        Update history after slicing a dataset.
        """
        for header in slicer.headers:
            if (header, keys) not in self.history['slicers'][slicer.category]:
                # Give it the next index
                self.history['slicers'][slicer.category][(header, keys)] = len(
                    self.history['slicers'][slicer.category]
                )

    def stow(self,
             cached_ops: Dict[CachedOperation, List[List[str]]],
             load_from_cache_file: bool = True):
        """
        Apply a list of cached operations in sequence.
        """

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
            "-".join(
                "-".join(str(k) + "-" + str(v) for k, v in f.items()) for f in self._data_files
            )
        )

        # Map the cached operations over the dataset
        dataset = self.map(_map_fn,
                           batched=True,
                           batch_size=32,
                           cache_file_name='cache-' + str(abs(val)) + '.arrow',
                           load_from_cache_file=load_from_cache_file)

        # TODO(karan): should this operation be in-place or return a mapped Dataset
        # self.__dict__.update(dataset.__dict__)
        # return self
        return dataset

    def slice(self, slicer, keys):
        """
        Slice the dataset.
        """
        # Update the history
        self.update_history(slicer, keys)
        return slicer(self, keys=keys)
