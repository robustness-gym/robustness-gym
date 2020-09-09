import json
from typing import List

import spacy

from robustness_gym.cached_ops.cached_ops import CachedOperation


class Spacy(CachedOperation):

    def __init__(self,
                 name: str = "en_core_web_sm"):
        super(Spacy, self).__init__(
            lang=name,
        )

        # Load up the spacy module
        spacy.prefer_gpu()
        self.name = name
        self._nlp = spacy.load(name)

    @property
    def nlp(self):
        return self._nlp

    @classmethod
    def encode(cls, obj) -> str:
        # Convert the Spacy Doc to json before caching
        return json.dumps(obj.to_json())

    def apply(self, text_batch: List[str]) -> List:
        # Apply spacy's pipe method to process the examples
        return list(self.nlp.pipe(text_batch))

    @classmethod
    def tokens(cls,
               decoded_batch: List) -> List[List[str]]:
        """
        For each example, returns the list of tokens extracted by Spacy for each key.

        Spacy stores the span of each token under the "tokens" key.
        This function extracts the tokens from the text using the span of each token.
        """

        token_batch = []
        # Iterate over each decoded Doc dictionary
        for doc_dict in decoded_batch:
            tokens = []
            for token_dict in doc_dict['tokens']:
                tokens.append(doc_dict['text'][token_dict['start']:token_dict['end']])

            token_batch.append(tokens)

        return token_batch
