from __future__ import annotations

import json
from json import JSONDecodeError

from mosaic import DataPanel

from robustnessgym.core.constants import CURATION
from robustnessgym.core.identifier import Identifier


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
