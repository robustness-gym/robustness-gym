from __future__ import annotations

import json
from json import JSONDecodeError

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
            self._identifier = dataset.identifier
            self.lineage = [(str(Dataset.__name__), dataset.identifier)]
        else:
            super(Slice, self).__init__(*args, **kwargs)

            # Set the identifier
            self._identifier = identifier

            # A slice has a lineage
            self.lineage = []

        # Always a single slice inside a slice
        self.num_slices = 1

        # Set the category of the slice: defaults to 'curated'
        self.category = CURATION

    @property
    def identifier(self):
        if self._identifier:
            return self._identifier
        if self.lineage:
            short_lineage = []
            for entry in self.lineage:
                if len(entry) == 3:
                    try:
                        columns = json.loads(entry[2])
                    except JSONDecodeError:
                        columns = entry[2]
                    short_lineage.append(str(entry[1]) + " @ " + str(columns))
                else:
                    short_lineage.append(str(entry[1]))
            self._identifier = Identifier(_name=" -> ".join(short_lineage))
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
