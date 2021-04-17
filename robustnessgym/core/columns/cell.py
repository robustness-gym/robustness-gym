import os
from collections import Sequence

import dill
import yaml

from robustnessgym.core.cells.abstract import AbstractCell
from robustnessgym.core.cells.cell import Cell


class CellColumn:
    def __init__(self, cells: Sequence[AbstractCell]):
        self.cells = cells

    def __getitem__(self, index):
        return self.cells[index]

    def write(self, path: str, write_together: bool = False) -> None:

        if write_together:
            # Make directory
            os.makedirs(path, exist_ok=True)

            # Get the paths where metadata and data should be stored
            metadata_path = os.path.join(path, "meta.yaml")
            data_path = os.path.join(path, "data.pkl")

            # Saving the metadata as a yaml
            yaml.dump(
                {
                    "dtype": type(self),
                    "cell_dtypes": list(map(type, self)),
                    "len": len(self),
                },
                metadata_path,
            )

            # Saving all cell data in a single pickle file
            dill.dump([cell.encode() for cell in self.cells], open(data_path, "wb"))
        else:
            os.makedirs(path, exist_ok=True)
            dill.dump(map(type, self.cells), open(os.path.join(path, "types.pkl")))
            # Save all the cells separately
            [cell.write(path) for cell in self.cells]

    @classmethod
    def read(cls, path: str, cell_type, index) -> Cell:
        # Read in the metadata
        meta = yaml.read(os.path.join(path, "meta.yaml"), Loader=yaml.FullLoader)
        # Read in the actual cells
        cells = dill.load(os.path.join(path, "data.pkl"))
        # Decode all the cells
        cells = [
            cell_dtype.decode(cell)
            for cell_dtype, cell in zip(meta["cell_dtype"], cells)
        ]

        return cls(
            cells,
        )
