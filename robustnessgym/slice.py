from __future__ import annotations

from robustnessgym.dataset import Dataset


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
        self.identifier = identifier

        # Always a single slice inside a slice
        self.num_slices = 1

        # A slice has a lineage
        self.lineage = None

        # Set the category of the slice: defaults to 'curated'
        self.category = 'curated'

    @classmethod
    def from_dataset(cls,
                     dataset: Dataset,
                     identifier: str):
        return cls(identifier=identifier, dataset=dataset)
    #
    # def __repr__(self):
    #     schema_str = dict((a, str(b)) for a, b in zip(self._data.schema.names, self._data.schema.types))
    #     return f"{self.__class__.__name__}(schema: {schema_str}, num_rows: {self.num_rows})"
