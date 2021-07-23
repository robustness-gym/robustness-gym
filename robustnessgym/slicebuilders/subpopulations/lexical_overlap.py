from typing import List

import numpy as np

from robustnessgym.core.operation import lookup
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.ops.spacy import SpacyOp
from robustnessgym.slicebuilders.subpopulations.score import ScoreSubpopulation


class LexicalOverlapSubpopulation(ScoreSubpopulation):
    def score(
        self,
        batch: DataPanel,
        columns: List[str],
        *args,
        **kwargs,
    ) -> np.ndarray:
        # Require that the number of keys is exactly 2
        assert len(columns) == 2, "Must specify exactly 2 keys."

        # Lookup the tokens after lower-casing and placing into a set
        try:
            tokens = {
                col: [
                    set([str(tok).lower() for tok in doc])
                    for doc in lookup(batch, SpacyOp, [col])
                ]
                for col in columns
            }
        except AttributeError:
            tokens = {
                col: [
                    set([str(tok).lower() for tok in text.split()])
                    for text in batch[col]
                ]
                for col in columns
            }

        # Compute the intersection over union score
        return np.array(
            [
                len(tokens_0.intersection(tokens_1))
                / float(len(tokens_0.union(tokens_1)))
                for tokens_0, tokens_1 in zip(tokens[columns[0]], tokens[columns[1]])
            ]
        )
