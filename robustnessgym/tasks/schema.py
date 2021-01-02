from typing import Callable, Collection, Dict, OrderedDict

from datasets.features import FeatureType

from robustnessgym.core.tools import get_all_paths


class Schema:
    def __init__(
        self, features: OrderedDict, grounding_candidates: Dict[str, Collection]
    ):
        # Store the features and grounding candidates
        self.features = features
        self.grounding_candidates = grounding_candidates
        self.reversed_grounding_candidates = {
            v: k for k, values in self.grounding_candidates.items() for v in values
        }

    def ground(self, features: Dict[str, FeatureType]):
        """

        Args:
            features: given by Dataset.features

        Returns: (grounding, reversed_grounding)

        """
        # For features, get the path to the leaves in the (potentially nested)
        # features dictionary
        flat_columns = get_all_paths(features)
        flat_columns = {
            tuple(path) if len(path) > 1 else path[0] for path in flat_columns
        }

        # Figure out the (reversed) grounding: map columns in the dataset to keys in
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
        # TODO(karan): if not, add code to automatically rejig the dataset in map_fn
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

    def keys(self):
        return list(self.features.keys())
