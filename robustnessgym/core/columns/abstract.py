from __future__ import annotations
from abc import abstractmethod
from typing import Callable, List, Optional, Sequence

from robustnessgym.core.identifier import Identifier
import numpy as np


class AbstractCell:
    def __init__(self):
        pass

    @abstractmethod
    def materialize(self):
        raise NotImplementedError
    
    @property
    def is_materialized(self):
        return self.data is not None 


class AbstractColumn:
    def __init__(self, num_rows: int, identifier: Identifier = None):

        self._identifier = Identifier("column") if not identifier else identifier

        self.index = [str(i) for i in range(num_rows)]
        self._materialized = False 

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
        return cls(data=cells)

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

    def update(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        **kwargs,
    ) -> AbstractColumn:
        raise NotImplementedError

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
