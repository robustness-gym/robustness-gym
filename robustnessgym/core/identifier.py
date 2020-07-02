"""Identifiers for objects in Robustness Gym."""
from __future__ import annotations

import json
from typing import Callable, List, Union

from robustnessgym.core.tools import persistent_hash


class Identifier:
    """Class for creating identifiers for objects in Robustness Gym."""

    def __init__(self, _name: str, _index: Union[str, int] = None, **kwargs):

        self._name = _name
        self._index = str(_index) if _index is not None else None
        self._parameters = kwargs

        for param, value in self.parameters.items():
            if isinstance(value, Callable):
                self.parameters[param] = ".".join(
                    [str(value.__module__), str(value.__name__)]
                )
            else:
                self.parameters[param] = str(value)

    @property
    def name(self):
        return self._name

    @property
    def index(self):
        return self._index

    @property
    def parameters(self):
        return self._parameters

    @classmethod
    def range(cls, n: int, _name: str, **kwargs) -> List[Identifier]:

        if n > 1:
            return [cls(_name=_name, _index=i, **kwargs) for i in range(1, n + 1)]
        return [cls(_name=_name, **kwargs)]

    def __repr__(self):
        params = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        if self.index is not None:
            return (
                f"{self.name}-{self.index}({params})"
                if len(params) > 0
                else f"{self.name}-{self.index}"
            )
        return f"{self.name}({params})" if len(params) > 0 else f"{self.name}"

    def __hash__(self):
        return persistent_hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def dumps(self):
        return json.dumps(self.__dict__)

    @classmethod
    def loads(cls, s: str):
        identifier = Identifier(_name="")
        identifier.__dict__ = json.loads(s)
        return identifier
