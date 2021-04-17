from __future__ import annotations

import abc
from abc import abstractmethod


class AbstractCell(abc.ABC):
    data: object = None
    loader: object = None

    def __init__(self, *args, **kwargs):
        super(AbstractCell, self).__init__(*args, **kwargs)

    @abstractmethod
    def default_loader(self, *args, **kwargs):
        raise NotImplementedError("Must implement `default_loader`.")

    @abstractmethod
    def get(self, *args, **kwargs):
        """Get me the thing that this cell exists for."""
        raise NotImplementedError("Must implement `get`.")

    def encode(self):
        raise NotImplementedError("Must implement `encode`.")

    def decode(self):
        raise NotImplementedError("Must implement `decode`.")

    def write(self, path: str) -> None:
        raise NotImplementedError("Must implement `write`.")

    @classmethod
    def read(cls, path: str, *args, **kwargs) -> AbstractCell:
        raise NotImplementedError("Must implement `read`.")
