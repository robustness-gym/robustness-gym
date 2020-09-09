from typing import List, Dict, Union, Optional

import textblob

from robustness_gym.cached_ops.cached_ops import CachedOperation
from robustness_gym.identifier import Identifier


class TextBlob(CachedOperation):

    def __init__(self):
        # TODO(karan): requires running `python -m textblob.download_corpora`
        super(TextBlob, self).__init__()

    @classmethod
    def encode(cls, obj: textblob.TextBlob) -> str:
        # Dump the TextBlob object to JSON
        # This loses a lot of information
        # Unfortunately, TextBlob provides no way to serialize/deserialize objects
        return obj.to_json()

    @classmethod
    def retrieve(cls,
                 batch: Dict[str, List],
                 keys: List[str],
                 identifier: Union[str, Identifier] = None,
                 reapply: bool = False,
                 **kwargs) -> Optional[Dict[str, List]]:
        # Default to reapplying the TextBlob op when retrieving
        return super().retrieve(batch, keys, identifier, reapply=True, **kwargs)

    def apply(self,
              text_batch: List[str]) -> List:
        # Create a TextBlob for each example
        return [textblob.TextBlob(text) for text in text_batch]
