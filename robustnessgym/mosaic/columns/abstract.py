from __future__ import annotations

import abc
import logging
from abc import abstractmethod
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from robustnessgym.core.identifier import Identifier
from robustnessgym.mosaic.mixins.collate import CollateMixin
from robustnessgym.mosaic.mixins.copying import CopyMixin
from robustnessgym.mosaic.mixins.identifier import IdentifierMixin
from robustnessgym.mosaic.mixins.index import IndexableMixin
from robustnessgym.mosaic.mixins.inspect_fn import FunctionInspectorMixin
from robustnessgym.mosaic.mixins.materialize import MaterializationMixin
from robustnessgym.mosaic.mixins.state import StateDictMixin
from robustnessgym.mosaic.mixins.storage import ColumnStorageMixin

logger = logging.getLogger(__name__)


class AbstractColumn(
    CollateMixin,
    ColumnStorageMixin,
    CopyMixin,
    FunctionInspectorMixin,
    IdentifierMixin,
    IndexableMixin,
    MaterializationMixin,
    StateDictMixin,
    abc.ABC,
):
    """An abstract class for Mosaic columns."""

    visible_rows: Optional[np.ndarray] = None
    _data: Sequence = None

    def __init__(
        self,
        num_rows: int,
        identifier: Identifier = None,
        *args,
        **kwargs,
    ):
        super(AbstractColumn, self).__init__(
            n=num_rows,
            identifier=identifier,
            *args,
            **kwargs,
        )

        # Log creation
        logger.info(f"Created `{self.__class__.__name__}` with {len(self)} rows.")

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return {}

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {"_materialize", "_collate_fn", "_data"}

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        # indices that return a single cell
        if isinstance(index, int) or isinstance(index, np.int):
            data = self.data[index]

            # check if the column implements materialization
            if self.materialize:
                return data.get()
            else:
                return data

        # indices that return batches
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            data = self.data[index]
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            data = [self.data[i] for i in index]
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            data = [self.data[int(i)] for i in index]
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

        if self.materialize:
            # if materializing, return a batch (by default, a list of objects returned
            # by `element.get`, otherwise the batch format specified by `self.collate`
            return self.collate([element.get() for element in data])
        else:
            # if not materializing, return a new Column
            return self.__class__(data, materialize=self.materialize)

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

    def __len__(self):
        # If only a subset of rows are visible
        if self.visible_rows is not None:
            return len(self.visible_rows)

        # Length of the underlying data stored in the column
        if self.data is not None:
            return len(self.data)
        return 0

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
        """Map a function over the elements of the column."""
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
        """Filter the elements of the column using a function."""
        raise NotImplementedError

    def set_visible_rows(self, indices: Optional[Sequence]):
        """Set the visible rows of the column."""
        if indices is None:
            self.visible_rows = None
        else:
            if len(indices):
                assert min(indices) >= 0 and max(indices) < len(self), (
                    f"Ensure min index {min(indices)} >= 0 and "
                    f"max index {max(indices)} < {len(self)}."
                )
            if self.visible_rows is not None:
                self.visible_rows = self.visible_rows[np.array(indices, dtype=int)]
            else:
                self.visible_rows = np.array(indices, dtype=int)

    def batch(
        self,
        batch_size: int = 32,
        drop_last_batch: bool = False,
        collate: bool = True,
        *args,
        **kwargs,
    ):
        """Batch the column.

        Args:
            batch_size: integer batch size
            drop_last_batch: drop the last batch if its smaller than batch_size
            collate: whether to collate the returned batches

        Returns:
            batches of data
        """
        if self.materialize:
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                collate_fn=self.collate if collate else lambda x: x,
                drop_last=drop_last_batch,
                *args,
                **kwargs,
            )
        else:
            for i in range(0, len(self), batch_size):
                if drop_last_batch and i + batch_size > len(self):
                    continue
                yield self[i : i + batch_size]
