from __future__ import annotations

import abc
import os
from abc import abstractmethod

import dill
import yaml
from yaml.representer import Representer

Representer.add_representer(abc.ABCMeta, Representer.represent_name)


class AbstractCell(abc.ABC):
    data: object = None
    loader: object = None

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super(AbstractCell, self).__init__(*args, **kwargs)

    @abstractmethod
    def default_loader(self, *args, **kwargs):
        raise NotImplementedError("Must implement `default_loader`.")

    @abstractmethod
    def get(self, *args, **kwargs):
        """Get me the thing that this cell exists for."""
        raise NotImplementedError("Must implement `get`.")

    def metadata(self) -> dict:
        return {}

    def encode(self):
        """Encode `self` in order to specify what information is important to
        store.

        By default, we just return `self` so the entire object is
        stored. For complex objects (e.g. Spacy Doc), we may want to
        return a compressed representation of the object here.
        """
        return self

    def write(self, path: str) -> None:
        """Actually write the encoded object to disk."""
        # Create the paths
        os.makedirs(path, exist_ok=True)
        data_path = os.path.join(path, "data.dill")
        metadata_path = os.path.join(path, "meta.yaml")

        encoded_self = self.encode()
        yaml.dump(
            {
                "dtype": type(self),
                **self.metadata(),
            },
            open(metadata_path, "w"),
        )
        return dill.dump(encoded_self, open(data_path, "wb"))

    @classmethod
    def decode(cls, encoding) -> AbstractCell:
        """Recover the object from its compressed representation.

        By default, we don't change the encoding.
        """
        return encoding

    @classmethod
    def read(cls, path: str, *args, **kwargs) -> AbstractCell:
        """Read the cell from disk."""
        assert os.path.exists(path), f"`path` {path} does not exist."

        # Create the paths
        data_path = os.path.join(path, "data.dill")
        metadata_path = os.path.join(path, "meta.yaml")

        metadata = dict(yaml.load(open(metadata_path, "r"), Loader=yaml.FullLoader))
        return metadata["dtype"].decode(dill.load(open(data_path, "rb")))

    @classmethod
    def read_metadata(cls, path: str, *args, **kwargs) -> dict:
        """Lightweight alternative to full read."""
        meta_path = os.path.join(path, "meta.yaml")
        return dict(yaml.load(open(meta_path, "r"), Loader=yaml.FullLoader))
