from typing import Callable


class Identifier:

    def __init__(self, name, **kwargs):
        self.name = name
        self.parameters = kwargs

        for param, value in self.parameters.items():
            if isinstance(value, Callable):
                self.parameters[param] = ".".join([str(value.__module__), str(value.__name__)])

    def __repr__(self):
        params = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{self.name}({params})"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)
