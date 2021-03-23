from __future__ import annotations

import json
from typing import List, Union

import cytoolz as tz

from robustnessgym.core.constants import (
    ATTACK,
    CACHEDOPS,
    SLICEBUILDERS,
    SUBPOPULATION,
    TRANSFORMATION,
)
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import persistent_hash, strings_as_json


class InteractionTape:
    """Class for interaction tape to keep track of applied ops."""

    def __init__(self):
        # Keep track of the history
        self.history = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(interactions={len(self.history)})"

    def __hash__(self):
        val = 0
        for (identifier, json_columns) in self.history:
            val ^= persistent_hash(str(identifier) + str(json_columns))
        return val

    def dumps(self):
        return json.dumps(
            {
                json.dumps((identifier.dumps(), json_columns)): idx
                for (identifier, json_columns), idx in self.history.items()
            }
        )

    @classmethod
    def loads(cls, s: str):
        tape = InteractionTape()
        history = json.loads(s)
        history = {
            tuple(json.loads(json_tuple)): idx for json_tuple, idx in history.items()
        }
        tape.history = {
            (Identifier.loads(identifier), json_columns): idx
            for (identifier, json_columns), idx in history.items()
        }

        return tape

    def update(self, identifier: Union[str, Identifier], columns: List[str]) -> None:
        """Update the interaction tape with information about an interaction.

        Args:
            identifier: Identifier for the interaction used.
            columns: list of columns on which the interaction was applied.

        Returns: True if the interaction was added to the tape, False if it was
        already applied before.
        """
        if isinstance(identifier, str):
            identifier = Identifier(_name=identifier)
        elif isinstance(identifier, Identifier):
            pass
        else:
            raise ValueError(
                f"Parameter `identifier` should be an instance of class Identifier "
                f"or str, "
                f"not {type(identifier)}."
            )

        # Dump the column names to JSON
        json_columns = strings_as_json(strings=columns)

        # Check if the entry is not in the history
        if (identifier, json_columns) not in self.history:
            # Give it the next index
            self.history[(identifier, json_columns)] = len(self.history)

    def check(self, identifier: Union[str, Identifier], columns: List[str]) -> bool:
        """

        Args:
            identifier:
            columns:

        Returns:

        """
        if not (isinstance(identifier, str) or isinstance(identifier, Identifier)):
            raise ValueError(
                f"Parameter `identifier` should be an instance of class Identifier "
                f"or str, "
                f"not {type(identifier)}."
            )

        # Dump the column names to JSON
        json_columns = strings_as_json(strings=columns)

        # Check if the entry is already in the history
        if (identifier, json_columns) in self.history:
            return True
        return False


class InteractionTapeHierarchyMixin:
    def __init__(self):
        self.interactions = {
            CACHEDOPS: InteractionTape(),
            SLICEBUILDERS: {
                SUBPOPULATION: InteractionTape(),
                TRANSFORMATION: InteractionTape(),
                ATTACK: InteractionTape(),
            },
        }

    def hash_interactions(self):
        v = 0
        for path in [
            [CACHEDOPS],
            [SLICEBUILDERS, SUBPOPULATION],
            [SLICEBUILDERS, TRANSFORMATION],
            [SLICEBUILDERS, ATTACK],
        ]:
            v ^= self.fetch_tape(path=path).__hash__()
        return v

    def dumps_interactions(self):
        return json.dumps(
            {
                CACHEDOPS: self.interactions[CACHEDOPS].dumps(),
                SLICEBUILDERS: {
                    SUBPOPULATION: self.interactions[SLICEBUILDERS][
                        SUBPOPULATION
                    ].dumps(),
                    TRANSFORMATION: self.interactions[SLICEBUILDERS][
                        TRANSFORMATION
                    ].dumps(),
                    ATTACK: self.interactions[SLICEBUILDERS][ATTACK].dumps(),
                },
            }
        )

    @classmethod
    def loads_interactions(cls, s: str) -> InteractionTapeHierarchyMixin:
        tape_hierarchy = InteractionTapeHierarchyMixin()
        interactions = json.loads(s)
        tape_hierarchy.interactions = {
            CACHEDOPS: InteractionTape.loads(interactions[CACHEDOPS]),
            SLICEBUILDERS: {
                SUBPOPULATION: InteractionTape.loads(
                    interactions[SLICEBUILDERS][SUBPOPULATION]
                ),
                TRANSFORMATION: InteractionTape.loads(
                    interactions[SLICEBUILDERS][TRANSFORMATION]
                ),
                ATTACK: InteractionTape.loads(interactions[SLICEBUILDERS][ATTACK]),
            },
        }
        return tape_hierarchy

    def update_tape(
        self,
        path: List[str],
        identifiers: Union[Identifier, List[Identifier]],
        columns: List[str],
    ):
        """Update the tape.

        Args:
            path: Location of the InteractionTape in the hierarchy.
            identifiers:
            columns:

        Returns:
        """
        # Fetch the tape
        tape = self.fetch_tape(path=path)

        # Update it
        if isinstance(identifiers, Identifier) or isinstance(identifiers, str):
            return tape.update(identifier=identifiers, columns=columns)
        else:
            return [
                tape.update(identifier=identifier, columns=columns)
                for identifier in identifiers
            ]

    def check_tape(
        self,
        path: List[str],
        identifiers: Union[Identifier, List[Identifier]],
        columns: List[str],
    ):
        """Check the tape.

        Args:

            path:
            identifiers:
            columns:

        Returns:
        """
        # Fetch the tape
        tape = self.fetch_tape(path=path)

        # Check it
        if isinstance(identifiers, Identifier) or isinstance(identifiers, str):
            return tape.check(identifier=identifiers, columns=columns)
        else:
            return [
                tape.check(identifier=identifier, columns=columns)
                for identifier in identifiers
            ]

    def fetch_tape(self, path: List[str]) -> InteractionTape:
        """Fetch an InteractionTape."""
        return tz.get_in(path, self.interactions)
