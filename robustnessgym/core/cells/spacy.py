from __future__ import annotations

import dill
from spacy.tokens import Doc

from robustnessgym.core.cells.abstract import AbstractCell


class SpacyCell(AbstractCell):
    def __init__(self, doc: Doc):
        # Put some data into the cell
        self.doc = doc

    def default_loader(self):
        pass

    def get(self):
        """Get me the data for this cell."""
        return self.doc

    def _encode(self):
        return

    @classmethod
    def _decode(self):
        return

    def write(self, path: str) -> None:
        # Custom thing
        dill.dump(self._encode(), open(path, "wb"))

    @classmethod
    def read(cls, path: str) -> SpacyCell:
        return cls._decode(dill.load(open(path, "rb")))
