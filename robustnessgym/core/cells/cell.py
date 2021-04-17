from __future__ import annotations

import dill

from robustnessgym.core.cells.abstract import AbstractCell


class Cell(AbstractCell):
    def __init__(self, data: object):
        # Put some data into the cell
        self.data = data

    def default_loader(self):
        pass

    def get(self):
        """Get me the data for this cell."""
        return self.data

    def write(self, path: str) -> None:
        dill.dump(self, open(path, "wb"))

    @classmethod
    def read(cls, path: str) -> Cell:
        return dill.load(open(path, "rb"))
