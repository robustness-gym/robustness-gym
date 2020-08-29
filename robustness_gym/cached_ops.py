from __future__ import annotations

import hashlib
import json
from functools import partial
from typing import *

import spacy
import torch
from allennlp.predictors import Predictor
from quinine.common.utils import rmerge

from robustness_gym import Dataset


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
        super(Spacy, self).__init__(identifier='spacy')

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
            identifier='allen_constituency_parser')

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
        super(StripText, self).__init__(identifier='strip_text')

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
        super(TextBlob, self).__init__(identifier='textblob')

    def apply(self, text_batch) -> List:
        pass
