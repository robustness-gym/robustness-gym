from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Callable, Dict, List, Optional, Union

from meerkat import AbstractColumn, DataPanel

from robustnessgym.core.constants import CURATION, GENERIC, SUBPOPULATION
from robustnessgym.core.identifier import Id, Identifier


class SliceMixin:
    """Slice class in Robustness Gym."""

    def __init__(self):
        # A slice has a lineage
        if self.identifier is None:
            self.lineage = []
        else:
            self.lineage = [(str(self.__class__.__name__), self.identifier)]

        # Set the category of the slice: defaults to 'curated'
        self.category = CURATION

    def add_to_lineage(self, category, identifier, columns=None):
        """Append to the lineage."""
        # TODO (karan): add Identifier directly
        if columns:
            self.lineage.append((category, identifier, columns))
        else:
            self.lineage.append((category, identifier))

        # Update the identifier
        self._lineage_to_identifier()

    def _add_op_to_lineage(self):
        if self.node.last_parent is not None:
            opnode, indices = self.node.last_parent
            try:
                fn = opnode.captured_args["function"]
            except KeyError:
                return

            if opnode.ref().__name__ == "filter":
                self.add_to_lineage(
                    SUBPOPULATION.capitalize(),
                    Id("Function", name=fn.__name__, mem=hex(id(fn))),
                    [],
                )
                self.category = SUBPOPULATION
            else:
                self.add_to_lineage(
                    GENERIC.capitalize(),
                    Id("Function", name=fn.__name__, mem=hex(id(fn))),
                    [],
                )

    def _lineage_to_identifier(self):
        """Synchronize to the current lineage by reassigning to
        `self._identifier`."""
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
        # Assign the new lineage to the identifier
        self._identifier = Identifier(_name=" -> ".join(short_lineage))

    @property
    def identifier(self):
        """Slice identifier."""
        if self._identifier:
            return self._identifier
        if self.lineage:
            self._lineage_to_identifier()
            return self._identifier
        return None

    @identifier.setter
    def identifier(self, value):
        """Set the slice's identifier."""
        self._identifier = value

    @classmethod
    def _add_state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {
            "lineage",
            "category",
        }


class SliceDataPanel(DataPanel, SliceMixin):
    def __init__(self, *args, **kwargs):
        super(SliceDataPanel, self).__init__(*args, **kwargs)
        SliceMixin.__init__(self)

    @classmethod
    def _state_keys(cls) -> set:
        state_keys = super(SliceDataPanel, cls)._state_keys()
        state_keys.union(cls._add_state_keys())
        return state_keys

    def update(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        remove_columns: Optional[List[str]] = None,
        num_workers: int = 0,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ) -> SliceDataPanel:
        dp = super(SliceDataPanel, self).update(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            remove_columns=remove_columns,
            num_workers=num_workers,
            materialize=materialize,
            pbar=pbar,
            **kwargs,
        )
        if isinstance(dp, SliceDataPanel):
            dp._add_op_to_lineage()

        return dp

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ) -> Optional[SliceDataPanel]:
        dp = super(SliceDataPanel, self).filter(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_workers=num_workers,
            materialize=materialize,
            pbar=pbar,
            **kwargs,
        )
        if isinstance(dp, SliceDataPanel):
            dp._add_op_to_lineage()
        return dp

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        output_type: type = None,
        mmap: bool = False,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ) -> Optional[Union[Dict, List, AbstractColumn]]:
        dp = super(SliceDataPanel, self).map(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_workers=num_workers,
            output_type=output_type,
            mmap=mmap,
            materialize=materialize,
            pbar=pbar,
            **kwargs,
        )
        if isinstance(dp, SliceDataPanel):
            dp._add_op_to_lineage()
        return dp
