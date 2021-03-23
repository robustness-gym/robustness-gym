from __future__ import annotations

import json
from copy import copy
from json import JSONDecodeError

from robustnessgym.core.constants import CURATION
from robustnessgym.core.dataset import Dataset
from robustnessgym.core.identifier import Identifier


class Slice(Dataset):
    """Slice class in Robustness Gym."""

    def __init__(
        self,
        *args,
        identifier: Identifier = None,
        **kwargs,
    ):
        # Set the identifier
        self._identifier = identifier

        # A slice has a lineage
        self.lineage = []

        # Set the category of the slice: defaults to 'curated'
        self.category = CURATION

        # Create a Slice directly from a Dataset
        if len(args) == 1 and isinstance(args[0], Dataset):
            # Update with the dataset info
            dataset = args[0]
            self.__dict__.update(dataset.__dict__)
            self._dataset = copy(dataset._dataset)
            # self._identifier = identifier or dataset.identifier
            self.lineage = [(str(Dataset.__name__), dataset.identifier)]
        else:
            super(Slice, self).__init__(*args, **kwargs)

    def __repr__(self):
        return (
            f"RG{self.__class__.__name__}["
            f"num_rows: {self.num_rows}]({self.identifier})"
        )

    def add_to_lineage(self, category, identifier, columns=None):
        """Append to the lineage."""
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
    def from_dataset(cls, dataset: Dataset):
        """Create a slice from a dataset."""
        return cls(dataset)

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        dataset_state_keys = super(Slice, cls)._state_keys()
        return dataset_state_keys.union(
            {
                "lineage",
                "category",
            }
        )

    def __getstate__(self):
        """Get the current state of the slice."""

        dataset_state = super(Slice, self).__getstate__()
        state = {
            **dataset_state,
            **{
                "lineage": [
                    tuple(t[:1])
                    + (t[1].dumps(),)
                    + (tuple(t[2:]) if len(t) > 2 else ())
                    for t in self.lineage
                ],
                "category": self.category,
            },
        }
        self._assert_state_keys(state)

        return state

    def __setstate__(self, state):
        """Set the current state of the slice."""
        # Check that the state contains all keys
        self._assert_state_keys(state)

        # Load the lineage
        self.lineage = [
            tuple(t[:1])
            + (Identifier.loads(t[1]),)
            + (tuple(t[2:]) if len(t) > 2 else ())
            for t in state["lineage"]
        ]

        # Load the category
        self.category = state["category"]

        # Set the dataset state: pick out state keys that correspond to the Dataset
        super(Slice, self).__setstate__(
            {k: v for k, v in state.items() if k in super(Slice, self)._state_keys()}
        )
