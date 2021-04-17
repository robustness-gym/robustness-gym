from __future__ import annotations

import abc
import logging
from abc import abstractmethod
from typing import Callable, List, Optional, Sequence

import numpy as np

from robustnessgym.core.cells.abstract import AbstractCell
from robustnessgym.core.identifier import Identifier

logger = logging.getLogger(__name__)


class AbstractColumn(abc.ABC):
    visible_rows: Optional[np.ndarray]
    _data: Sequence

    def __init__(self, num_rows: int, identifier: Identifier = None, *args, **kwargs):
        super(AbstractColumn, self).__init__(*args, **kwargs)

        # Identifier for the column
        self._identifier = Identifier("column") if not identifier else identifier

        # Index associated with each cell of the column
        self.index = [str(i) for i in range(num_rows)]

        # Whether data in the column is materialized
        self._materialized = False

        # Log creation
        logger.info(f"Created `{self.__class__.__name__}` with {len(self)} rows.")

    def __len__(self):
        return len(self._data)

    def _remap_index(self, index):
        if isinstance(index, int):
            return self.visible_rows[index].item()
        elif isinstance(index, slice):
            return self.visible_rows[index].tolist()
        elif isinstance(index, str):
            return index
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return self.visible_rows[index].tolist()
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            return self.visible_rows[index].tolist()
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        if isinstance(index, int) or isinstance(index, np.int):
            cell = self._data[index]
            if self._materialized:
                return cell.materialize()
            else:
                return cell

                # indices that return batches
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            cells = self._data[index]
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            cells = [self._data[i] for i in index]
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            cells = [self._data[int(i)] for i in index]
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

        if self._materialized:
            [cell.materialize() for cell in cells]

        return self.from_cells(cells)

    @classmethod
    def from_cells(cls, cells: List[AbstractCell]):
        assert cls != AbstractColumn, "Cannot run `from_cells` on abstract class."
        return cls(data=cells)

    # @abstractmethod
    # def __getitem__(self):
    #     raise NotImplementedError()

    @abstractmethod
    def __setitem__(self):
        raise NotImplementedError()

    # @abstractmethod
    # def __len__(self) -> int:
    #     raise NotImplementedError()

    def encode(self):
        return self

    @classmethod
    def decode(cls):
        raise NotImplementedError()

    @abstractmethod
    def write(self, path) -> None:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def read(cls) -> AbstractColumn:
        raise NotImplementedError()

    @abstractmethod
    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        **kwargs,
    ) -> AbstractColumn:
        raise NotImplementedError

    @abstractmethod
    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        **kwargs,
    ) -> AbstractColumn:
        raise NotImplementedError
