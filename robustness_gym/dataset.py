from __future__ import annotations

from functools import partial
from typing import *

import nlp
import spacy
import torch
from allennlp.predictors import Predictor
from pyarrow import json
from quinine.common.utils import rmerge


class PreprocessingMixin:
    # Set up AllenNLP's constituency parser
    if torch.cuda.is_available():
        constituency_parser = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
            cuda_device=0
        )
    else:
        constituency_parser = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
        )

    # Load up the spacy module
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")

    @classmethod
    def preprocess(cls, examples: Dict[List], keys):

        for key in keys:
            examples = cls.strip_text(examples, key)
            examples = cls.spacy_pipe(examples, key)
            examples = cls.constituency_parse(examples, key)

        return examples

    @classmethod
    def strip_text(cls, examples: Dict[List], key):
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
        return cls.update_cache(examples, [{'stripped': {key: val}} for val in stripped])

    @classmethod
    def spacy_pipe(cls, examples: Dict[List], key):

        # Apply spacy's pipe method to process the examples
        docs = list(cls.nlp.pipe(examples[key]))

        # Convert the docs to json and update the examples
        return cls.update_cache(examples, [{'spacy': {key: val.to_json()}} for val in docs])

    @classmethod
    def constituency_parse(cls, examples: Dict[List], key):

        # Apply the constituency parser
        parse_trees = cls.constituency_parser.predict_batch_json([{'sentence': example} for example in examples[key]])

        # Extract the tree from the output of the constituency parser and update the examples
        return cls.update_cache(examples, [{'constituency_parse': {key: val['trees']}} for val in parse_trees])

    @classmethod
    def update_cache(cls, examples: Dict[List], updates: List[Dict]) -> Dict[List]:
        """
        Updates the cache of preprocessed information stored with each example in a batch.

        examples must contain a key called 'cache' that maps to a dictionary.
        """
        assert 'cache' in examples, "Examples must have a key called 'cache'."
        assert len(examples['cache']) == len(updates), "Number of examples must equal the number of updates."

        # Update the cache
        # examples['cache'] is a list of dictionaries, one per example
        # updates is a list of dictionaries, one per example

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


class Dataset(nlp.Dataset, DatasetHelpersMixin, PreprocessingMixin):

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
        return f"Dataset(schema: {schema_str}, num_rows: {self.num_rows}, num_slices: {self.num_slices})"

    @classmethod
    def from_nlp(cls,
                 dataset: nlp.Dataset):
        """
        Create a Dataset from a Huggingface nlp.Dataset.
        """
        return Dataset(dataset)

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
            return dict(map(lambda t: (t[0], Dataset(t[1])), dataset.items()))
        else:
            return Dataset(dataset)

    @classmethod
    def from_json(cls,
                  json_path: str) -> Dataset:
        """
        Load a dataset from a JSON file on disk, where each line of the json file consists of a single example.
        """
        return Dataset(json.read_json(json_path))

    @classmethod
    def from_slice(cls):
        pass

    @classmethod
    def from_tfds(cls):
        # TODO(karan): v1 of robustness gym. Use it for image-based tasks, like clevr.
        pass

    def initialize(self):

        # For convenience
        dataset = self

        # Add an index to the dataset
        dataset = dataset.map(DatasetHelpersMixin.add_index, with_indices=True)

        # Add a dictionary for keeping track of slice membership
        dataset = dataset.map(DatasetHelpersMixin.add_slices_key)

        # Add a dictionary for keeping track of cached information
        dataset = dataset.map(DatasetHelpersMixin.add_cache_key)

        # Apply an expensive preprocessing step using Spacy
        dataset = dataset.map(partial(PreprocessingMixin.preprocess, keys=['question']), batched=True, batch_size=32)

        self.__dict__.update(dataset.__dict__)
