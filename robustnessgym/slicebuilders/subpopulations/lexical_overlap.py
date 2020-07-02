from typing import Dict, List

import cytoolz as tz
import numpy as np

from robustnessgym.cachedops.spacy import Spacy
from robustnessgym.slicebuilders.subpopulations.score import ScoreSubpopulation


class LexicalOverlapSubpopulation(ScoreSubpopulation):
    def score(
        self, batch: Dict[str, List], columns: List[str], *args, **kwargs
    ) -> np.ndarray:
        # Require that the number of keys is exactly 2
        assert len(columns) == 2, "Must specify exactly 2 keys."

        # Retrieve the tokens after lower-casing and placing into a set
        tokens = Spacy.retrieve(
            batch=batch,
            columns=[[key] for key in columns],
            proc_fns=tz.compose(
                # Lower case and put the tokens in a set for each tokenized text in
                # the batch
                lambda l: np.array(
                    [set([str(tok).lower() for tok in toks]) for toks in l]
                ),
                # Tokenize
                Spacy.tokens,
            ),
        )

        # Compute the intersection over union score
        return np.array(
            [
                len(tokens_0.intersection(tokens_1))
                / float(len(tokens_0.union(tokens_1)))
                for tokens_0, tokens_1 in zip(tokens[columns[0]], tokens[columns[1]])
            ]
        )
