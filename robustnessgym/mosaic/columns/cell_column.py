from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from tqdm.auto import tqdm

from robustnessgym.core.tools import convert_to_batch_column_fn
from robustnessgym.mosaic.cells.abstract import AbstractCell
from robustnessgym.mosaic.columns.abstract import AbstractColumn

logger = logging.getLogger(__name__)


class CellColumn(AbstractColumn):
    def __init__(
        self,
        cells: Sequence[AbstractCell] = None,
        materialize: bool = True,
        *args,
        **kwargs,
    ):
        self._data = cells
        self._materialize = materialize

        self.visible_rows = None

        super(CellColumn, self).__init__(num_rows=len(self), *args, **kwargs)

    @classmethod
    def from_cells(cls, cells: Sequence[AbstractCell], *args, **kwargs):
        return cls(cells=cells, *args, **kwargs)

    @property
    def data(self):
        return self._data

    @property
    def cells(self):
        return self._data

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = None,
        materialize: bool = None,
        **kwargs,
    ) -> Optional[Union[Dict, List]]:
        """Apply a map over the dataset."""
        # Check if need to materialize:
        # TODO(karan): figure out if we need materialize=False

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Ensure that num_proc is not None
        if num_proc is None:
            num_proc = 0

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_column_fn(function, with_indices=with_indices)
            batched = True
            logger.info(f"Converting `function` {function} to a batched function.")

        # # Get some information about the function
        # TODO: discuss whether this is actually required vs. doing it on first pass in
        # loop
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )

        # Run the map
        logger.info("Running `map`, the dataset will be left unchanged.")
        outputs = defaultdict(list) if function_properties.dict_output else []
        for i, batch in tqdm(
            enumerate(
                self.batch(
                    batch_size=batch_size,
                    drop_last_batch=drop_last_batch,
                    collate=batched,
                )
            ),
            total=(len(self) // batch_size)
            + int(not drop_last_batch and len(self) % batch_size != 0),
        ):

            # Run `function` on the batch
            output = (
                function(
                    batch,
                    range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                )
                if with_indices
                else function(batch)
            )

            # Append the output
            if output is not None:
                if isinstance(output, Mapping):
                    for k in output.keys():
                        outputs[k].extend(output[k])
                else:
                    outputs.extend(output)

        if not len(outputs):
            return None
        elif isinstance(outputs, dict):
            # turns the defaultdict into dict
            return dict(outputs)
        return outputs

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = 64,
        **kwargs,
    ) -> Optional[CellColumn]:
        """Apply a filter over the dataset."""
        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        # Get some information about the function
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new dataset will be returned.")
        outputs = self.map(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_proc=num_proc,
        )
        indices = np.where(outputs)[0]

        new_column = self.copy()
        new_column.set_visible_rows(indices)
        return new_column
