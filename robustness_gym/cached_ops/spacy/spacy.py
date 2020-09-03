from typing import List, Dict

import spacy

from robustness_gym.cached_ops.cached_ops import CachedOperation
from robustness_gym.tools import persistent_hash


class Spacy(CachedOperation):

    def __init__(self,
                 name: str = "en_core_web_sm"):
        super(Spacy, self).__init__(
            identifier=self.__class__.__name__,
        )

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

    def apply(self, text_batch: List[str]) -> List:
        # Apply spacy's pipe method to process the examples
        docs = list(self.nlp.pipe(text_batch))

        # Convert the docs to json
        return [val.to_json() for val in docs]

    @classmethod
    def get_tokens(cls,
                   batch: Dict[str, List],
                   keys: List[str]) -> List[Dict[str, List[str]]]:
        """
        For each example, returns the list of tokens extracted by spacy for each key.
        """
        return [
            {key: cls.tokens_from_spans(
                doc_json=cache['Spacy'][key]) for key in keys}
            for cache in batch['cache']
        ]

    @classmethod
    def tokens_from_spans(cls, doc_json: Dict) -> List[str]:
        """
        Spacy stores the span of each token under the "tokens" key.

        Use this function to actually extract the tokens from the text using the span of each token.
        """
        tokens = []
        for token_dict in doc_json['tokens']:
            tokens.append(doc_json['text']
                          [token_dict['start']:token_dict['end']])

        return tokens
