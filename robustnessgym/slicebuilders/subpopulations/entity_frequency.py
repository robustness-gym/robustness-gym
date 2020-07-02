from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from robustnessgym.cachedops.spacy import Spacy
from robustnessgym.core.dataset import Dataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.slicebuilders.subpopulation import Subpopulation


# TODO(karan): Inherit from MultiScoreSubpopulation
class EntityFrequency(Subpopulation, Spacy):
    def __init__(self, entity_thresholds: List[Tuple[str, List[int]]], *args, **kwargs):

        identifiers = []
        for entity_type, thresholds in entity_thresholds:
            for threshold in thresholds:
                identifiers.append(
                    Identifier(
                        _name=self.__class__.__name__,
                        entity_type=entity_type,
                        threshold=threshold,
                    )
                )

        super(EntityFrequency, self).__init__(identifiers, *args, **kwargs)

        if len(entity_thresholds) == 0:
            raise ValueError("At least one entity type required")

        for entity_type, _ in entity_thresholds:
            if entity_type not in [
                "PERSON",
                "NORP",
                "FAC",
                "ORG",
                "GPE",
                "LOC",
                "PRODUCT",
                "EVENT",
                "WORK_OF_ART",
                "LAW",
                "LANGUAGE",
                "DATE",
                "TIME",
                "PERCENT",
                "MONEY",
                "QUANTITY",
                "ORDINAL",
                "CARDINAL",
            ]:
                raise ValueError(f"Invalid entity type: {entity_type}")

        # List of tuples, each of which contains an entity type and a list of
        # associated frequency thresholds
        self.entity_thresholds = entity_thresholds

    @classmethod
    def from_dataset(
        cls, dataset: Dataset, entity_percentiles: List[Tuple[str, List[float]]]
    ) -> EntityFrequency:
        """Determine thresholds from dataset and specific percentiles."""
        raise NotImplementedError

    def apply(
        self,
        slice_membership: np.ndarray,
        batch: Dict[str, List],
        columns: List[str],
        *args,
        **kwargs,
    ) -> np.ndarray:

        if len(columns) != 1:
            raise ValueError("Only one key allowed")
        key = columns[0]

        for i, cache_item in enumerate(batch["cache"]):
            entities = cache_item["Spacy"][key]["ents"]
            entity_types = [ent["label"] for ent in entities]
            counts = Counter(entity_types)
            slice_ndx = 0
            for entity_type, thresholds in self.entity_thresholds:
                for threshold in thresholds:
                    if counts[entity_type] >= threshold:
                        slice_membership[i, slice_ndx] = 1
                    slice_ndx += 1

        return slice_membership
