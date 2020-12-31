from typing import List, Dict, Union, Optional

import textblob

from robustnessgym.core.cachedops import SingleColumnCachedOperation
from robustnessgym.core.identifier import Identifier


class TextBlob(SingleColumnCachedOperation):

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
    def retrieve(self_or_cls,
                 batch: Dict[str, List],
                 columns: List[str],
                 identifier: Union[str, Identifier] = None,
                 reapply: bool = False,
                 **kwargs) -> Optional[Dict[str, List]]:
        # Default to reapplying the TextBlob op when retrieving
        return super().retrieve(batch, columns, identifier, reapply=True, **kwargs)

    def single_column_apply(self,
                            column_batch: List,
                            *args,
                            **kwargs) -> List:
        # Create a TextBlob for each example
        return [textblob.TextBlob(text) for text in column_batch]
