from __future__ import annotations
from typing import Callable, Union, List


class Identifier:

    def __init__(self,
                 _name: str,
                 _index: Union[str, int] = None,
                 **kwargs):

        self._name = _name
        self._index = str(_index) if _index else _index
        self._parameters = kwargs

        for param, value in self.parameters.items():
            if isinstance(value, Callable):
                self.parameters[param] = ".".join([str(value.__module__), str(value.__name__)])

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
    def range(cls,
              n: int,
              _name: str,
              **kwargs) -> List[Identifier]:

        return [cls(
            _name=_name,
            _index=i,
            **kwargs) for i in range(n)
        ]

    def __repr__(self):
        params = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        if self.index:
            return f"{self.name}-{self.index}({params})"
        return f"{self.name}({params})"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)
