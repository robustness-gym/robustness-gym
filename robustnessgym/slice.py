from __future__ import annotations

from robustnessgym.identifier import Identifier
from robustnessgym.dataset import Dataset
from robustnessgym.constants import CURATION


class Slice(Dataset):

    def __init__(self,
                 *args,
                 identifier: str = None,
                 dataset: Dataset = None,
                 **kwargs):

        if dataset is not None:
            # Create a Slice directly from the Dataset object
            self.__dict__ = dataset.__dict__.copy()
        else:
            super(Slice, self).__init__(*args, **kwargs)

        # Set the identifier
        self._identifier = identifier

        # Always a single slice inside a slice
        self.num_slices = 1

        # A slice has a lineage
        self.lineage = []

        # Set the category of the slice: defaults to 'curated'
        self.category = CURATION

    @property
    def identifier(self):
        if self._identifier:
            return self._identifier
        if self.lineage:
            self._identifier = Identifier(_name=" -> ".join([str(entry[1]) for entry in self.lineage]))
            return self._identifier
        return None

    @identifier.setter
    def identifier(self, value):
        self._identifier = value

    @classmethod
    def from_dataset(cls,
                     dataset: Dataset,
                     identifier: str):
        return cls(identifier=identifier, dataset=dataset)

    def __repr__(self):
        return f"{self.__class__.__name__}[category: {self.category}, num_rows: {self.num_rows}]({self.identifier})"
