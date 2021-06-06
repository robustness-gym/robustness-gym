from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Collection, Dict, List, Tuple

from datasets.features import FeatureType

from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.core.tools import get_all_paths


class Schema:
    """Class for task input/output schemas in Robustness Gym."""

    def __init__(
        self,
        features: OrderedDict,
        grounding_candidates: Dict[str, Collection],
    ):
        # Store the features and grounding candidates
        self.features = features
        self.grounding_candidates = grounding_candidates
        self.reversed_grounding_candidates = {
            v: k for k, values in self.grounding_candidates.items() for v in values
        }

    @classmethod
    def from_columns(
        cls,
        features: Dict[str, FeatureType],
        columns: List[str],
    ) -> Schema:
        """Create a schema using features and columns."""
        for col in columns:
            assert col in features, f"Column {col} must be in `features`."

        return Schema(
            features=OrderedDict({k: v for k, v in features.items() if k in columns}),
            grounding_candidates={k: {k} for k in columns},
        )

    @classmethod
    def for_dataset(
        cls,
        dp: DataPanel,
        input_columns: List[str],
        output_columns: List[str],
    ) -> Tuple[Schema, Schema]:
        """Create input and output schemas using features, input and output
        columns."""
        # Set the features
        features = dp.features

        return Schema.from_columns(features, input_columns), Schema.from_columns(
            features, output_columns
        )

    def ground(self, features: Dict[str, FeatureType]):
        """

        Args:
            features: given by DataPanel.features

        Returns: (grounding, reversed_grounding)

        """
        # For features, get the path to the leaves in the (potentially nested)
        # features dictionary
        flat_columns = get_all_paths(features)
        flat_columns = {
            tuple(path) if len(path) > 1 else path[0] for path in flat_columns
        }

        # Figure out the (reversed) grounding: map columns in the dp to keys in
        # the schema
        reversed_grounding = {}
        for k in self.reversed_grounding_candidates:
            if ((isinstance(k, tuple) or isinstance(k, str)) and k in flat_columns) or (
                isinstance(k, Callable)
            ):
                reversed_grounding[k] = self.reversed_grounding_candidates[k]

        # Construct the grounding by reversing
        grounding = {v: k for k, v in reversed_grounding.items()}

        # Assert that the grounding covers the entire schema
        assert len(self.features) == len(grounding), "Grounding failed."

        # Assert that the grounded schema has the right types
        # FIXME(karan): Value == ClassLabel should be allowed: shouldn't break this
        # TODO(karan): if not, add code to automatically rejig the dp in map_fn
        # for key in self.features:
        #     if isinstance(grounding[key], str):
        #         assert self.features[key] == features[grounding[key]]
        #     elif isinstance(grounding[key], tuple):
        #         assert self.features[key] == tz.get_in(grounding[key], features)

        return grounding, reversed_grounding

    def __repr__(self):
        features = "\n\t".join([f"{k}: {v}" for k, v in self.features.items()])
        return f"Schema(\n\t{features}\n)"

    def __len__(self):
        return len(self.features)

    @property
    def columns(self):
        """List of columns that participate in the schema."""
        return list(self.features.keys())
