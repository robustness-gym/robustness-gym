from __future__ import annotations

import os
import pickle
from functools import partial
from typing import *

import nlp
import spacy
import torch
from allennlp.predictors import Predictor
from nlp.arrow_writer import ArrowWriter
from tqdm import tqdm
import pyarrow as pa
from pyarrow import json, table
from quinine.common.utils import rmerge
import cytoolz as tz
from copy import deepcopy


class PreprocessingMixin:

    def __init__(self):

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

        # Load up the spacy module
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess(self, examples: Dict[List], keys: List[str]):

        for key in keys:
            examples = self.strip_text(examples, key)
            examples = self.spacy_pipe(examples, key)
            examples = self.constituency_parse(examples, key)

        return examples

    def strip_text(self, examples: Dict[List], key):
        """
        A preprocessor that lower cases text and strips out punctuation.
        """

        # Clean up each text with a simple function
        stripped = list(
            map(
                lambda text: text.lower().replace(".", "").replace("?", "").replace("!", "").replace(",", ""),
                examples[key]
            )
        )

        # Update the examples with the stripped texts
        return self.update_cache(examples, [{'stripped': {key: val}} for val in stripped])

    def spacy_pipe(self, examples: Dict[List], key):

        # Apply spacy's pipe method to process the examples
        docs = list(self.nlp.pipe(examples[key]))

        # Convert the docs to json and update the examples
        return self.update_cache(examples, [{'spacy': {key: val.to_json()}} for val in docs])

    def constituency_parse(self, examples: Dict[List], key):

        # Apply the constituency parser
        parse_trees = self.constituency_parser.predict_batch_json([{'sentence': example} for example in examples[key]])

        # Extract the tree from the output of the constituency parser and update the examples
        return self.update_cache(examples, [{'constituency_parse': {key: val['trees']}} for val in parse_trees])

    def update_cache(self, examples: Dict[List], updates: List[Dict]) -> Dict[List]:
        """
        Updates the cache of preprocessed information stored with each example in a batch.

        - examples must contain a key called 'cache' that maps to a dictionary.
        - examples['cache'] is a list of dictionaries, one per example
        - updates is a list of dictionaries, one per example
        """
        assert 'cache' in examples, "Examples must have a key called 'cache'."
        assert len(examples['cache']) == len(updates), "Number of examples must equal the number of updates."

        # For each example, recursively merge the example's original cache dictionary with the update dictionary
        examples['cache'] = [rmerge(cache_dict, update_dict)
                             for cache_dict, update_dict in zip(examples['cache'], updates)]

        return examples


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
              PreprocessingMixin):

    def __init__(self, *args, **kwargs):

        if len(args) == 1 and isinstance(args[0], nlp.Dataset):
            # Create a Dataset directly from an nlp.Dataset object
            self.__dict__ = args[0].__dict__.copy()
        else:
            super(Dataset, self).__init__(*args, **kwargs)

        self.original_keys = self.schema.names
        self.num_slices = 0

    def __repr__(self):
        schema_str = dict((a, str(b)) for a, b in zip(self._data.schema.names, self._data.schema.types))
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
        return cls(json.read_json(json_path))

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
            return self.from_nlp(output)
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
        writer = ArrowWriter(schema=self.schema, path=os.path.join(path, 'data.arrow'), writer_batch_size=1000)

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

    def initialize(self):

        PreprocessingMixin.__init__(self)

        # For convenience
        dataset = self

        # Add an index to the dataset
        dataset = dataset.map(DatasetHelpersMixin.add_index, with_indices=True)

        # Add a dictionary for keeping track of slice membership
        dataset = dataset.map(DatasetHelpersMixin.add_slices_key)

        # Add a dictionary for keeping track of cached information
        dataset = dataset.map(DatasetHelpersMixin.add_cache_key)

        # Apply an expensive preprocessing step using Spacy
        dataset = dataset.map(partial(self.preprocess, keys=['question']), batched=True, batch_size=32)

        self.__dict__.update(dataset.__dict__)

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

    def slice(self, slicer):
        """
        Slice the dataset.
        """
        return slicer(self)
