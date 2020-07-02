from typing import List

from robustnessgym.core.cachedops import SingleColumnCachedOperation

try:
    import stanza
except ImportError:
    _stanza_available = False
else:
    _stanza_available = True


class Stanza(SingleColumnCachedOperation):
    """Class for running the Stanza pipeline using a CachedOperation.

    URL: https://stanfordnlp.github.io/stanza/
    """

    def __init__(self):
        if not _stanza_available:
            raise ImportError(
                "Stanza not available for import. Install using " "\npip install stanza"
            )
        super(Stanza, self).__init__()

        self._download()
        self.nlp = stanza.Pipeline()

    def _download(self):
        stanza.download()

    @classmethod
    def encode(cls, obj: stanza.Document) -> str:
        # Dump the Stanza Document to a string
        return obj.to_serialized()

    @classmethod
    def decode(cls, s: str):
        # Load the Stanza Document from the string
        return stanza.Document.from_serialized(s)

    def single_column_apply(self, column_batch: List, *args, **kwargs) -> List:
        # Create a doc for each example
        return [self.nlp(text) for text in column_batch]

    @classmethod
    def _get_attribute(
        cls, decoded_batch: List[stanza.Document], attribute: str
    ) -> List:
        """Get an arbitrary attribute using doc.get(attribute) from a list of
        Stanza Documents."""
        return [doc.get(attribute) for doc in decoded_batch]

    @classmethod
    def lemma(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of lemmatized words."""
        return cls._get_attribute(decoded_batch, "lemma")

    @classmethod
    def text(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of tokens."""
        return cls._get_attribute(decoded_batch, "text")

    @classmethod
    def upos(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of upos."""
        return cls._get_attribute(decoded_batch, "upos")

    @classmethod
    def xpos(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of xpos."""
        return cls._get_attribute(decoded_batch, "xpos")

    @classmethod
    def feats(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of feats."""
        return cls._get_attribute(decoded_batch, "feats")

    @classmethod
    def head(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of head."""
        return cls._get_attribute(decoded_batch, "head")

    @classmethod
    def deprel(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of deprel."""
        return cls._get_attribute(decoded_batch, "deprel")

    @classmethod
    def misc(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of misc."""
        return cls._get_attribute(decoded_batch, "misc")

    @classmethod
    def entities(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of entities."""
        return [doc.entities for doc in decoded_batch]

    @classmethod
    def id(cls, decoded_batch: List[stanza.Document]) -> List[List[str]]:
        """For each example, returns the list of ids."""
        return cls._get_attribute(decoded_batch, "id")
