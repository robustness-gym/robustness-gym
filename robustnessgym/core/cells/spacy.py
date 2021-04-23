from __future__ import annotations

try:
    import spacy
    from spacy.attrs import NAMES
    from spacy.tokens import Doc
    NAMES = [name for name in NAMES if name != "HEAD"]

    _is_spacy_available = True
except ImportError:
    _is_spacy_available = False
from robustnessgym.core.cells.abstract import AbstractCell


class SpacyCell(AbstractCell):
    def __init__(
        self,
        doc: Doc,
        *args,
        **kwargs,
    ):
        if not _is_spacy_available:
            raise ImportError("Please install spacy.")

        super(SpacyCell, self).__init__(*args, **kwargs)

        # Put some data into the cell
        self.doc = doc

    def default_loader(self, *args, **kwargs):
        return self

    def get(self, *args, **kwargs):
        return self.doc

    def get_state(self):
        arr = self.get().to_array(NAMES)
        return {
            "arr": arr.flatten(),
            "shape": list(arr.shape),
            "words": [t.text for t in self.get()],
        }

    @classmethod
    def from_state(cls, encoding, nlp: spacy.language.Language):
        doc = Doc(nlp.vocab, words=encoding["words"])
        return cls(doc.from_array(NAMES, encoding["arr"].reshape(encoding["shape"])))

    def __getitem__(self, index):
        return self.get()[index]

    def __getattr__(self, item):
        try:
            return getattr(self.get(), item)
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")

    def __repr__(self):
        return self.get().__repr__()


class LazySpacyCell(AbstractCell):
    def __init__(
        self,
        text: str,
        nlp: spacy.language.Langauge,
        *args,
        **kwargs,
    ):
        if not _is_spacy_available:
            raise ImportError("Please install spacy.")

        super(LazySpacyCell, self).__init__(*args, **kwargs)

        # Put some data into the cell
        self.text = text
        self.nlp = nlp

    def default_loader(self, *args, **kwargs):
        return self

    def get(self, *args, **kwargs):
        return self.nlp(self.text)

    def encode(self):
        arr = self.get().to_array(NAMES)
        return {
            "arr": arr.flatten(),
            "shape": list(arr.shape),
            "words": [t.text for t in self.get()],
        }

    @classmethod
    def decode(cls, encoding, nlp: spacy.language.Language):
        doc = Doc(nlp.vocab, words=encoding["words"])
        return doc.from_array(NAMES, encoding["arr"].reshape(encoding["shape"]))

    def __getitem__(self, index):
        return self.get()[index]

    def __getattr__(self, item):
        try:
            return getattr(self.get(), item)
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")

    def __repr__(self):
        return self.get().__repr__()
