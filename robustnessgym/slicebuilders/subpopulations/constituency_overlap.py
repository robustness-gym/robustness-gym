from typing import List

import fuzzywuzzy.fuzz as fuzz
import numpy as np
from meerkat.tools.lazy_loader import LazyLoader

from robustnessgym.core.decorators import prerequisites
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.operation import lookup
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.ops.allen import AllenConstituencyParsingOp
from robustnessgym.slicebuilders.subpopulations.score import ScoreSubpopulation

nltk = LazyLoader("nltk")


@prerequisites(AllenConstituencyParsingOp)
class ConstituencyOverlapSubpopulation(ScoreSubpopulation):
    def score(
        self,
        batch: DataPanel,
        columns: List[str],
        *args,
        **kwargs,
    ) -> np.ndarray:
        # Require that the number of keys is exactly 2
        assert len(columns) == 2, "Must specify exactly 2 keys."

        # Retrieve the trees
        trees = {
            col: lookup(batch, AllenConstituencyParsingOp, [col]) for col in columns
        }
        trees_0, trees_1 = trees[columns[0]], trees[columns[1]]

        # Fuzzy match the trees and return the `scores`
        return np.array(
            [
                fuzz.partial_token_set_ratio(
                    tree_0.replace("(", "").replace(")", "").replace(" ", ""),
                    tree_1.replace("(", "").replace(")", "").replace(" ", ""),
                )
                for tree_0, tree_1 in zip(trees_0, trees_1)
            ]
        )


@prerequisites(AllenConstituencyParsingOp)
class ConstituencySubtreeSubpopulation(ScoreSubpopulation):
    def __init__(self, *args, **kwargs):
        super(ConstituencySubtreeSubpopulation, self).__init__(
            intervals=[(1, 1)],
            identifiers=[Identifier(_name=self.__class__.__name__)],
            *args,
            **kwargs,
        )

    def score(
        self,
        batch: DataPanel,
        columns: List[str],
        *args,
        **kwargs,
    ) -> np.ndarray:
        # Require that the number of keys is exactly 2
        assert len(columns) == 2, "Must specify exactly 2 keys."

        # Retrieve the trees
        trees = {
            col: lookup(batch, AllenConstituencyParsingOp, [col]) for col in columns
        }
        trees_0, trees_1 = trees[columns[0]], trees[columns[1]]

        # Convert the trees corresponding to key 0 to NLTK trees
        trees_0 = [nltk.Tree.fromstring(tree) for tree in trees_0]

        # Find all subtrees of these trees
        all_subtrees_0 = [
            set(
                [
                    str(t).replace("\n", "").replace(" ", "").lower()
                    for t in tree_0.subtrees()
                ]
            )
            for tree_0 in trees_0
        ]

        # Output a score of 1 if the tree corresponding to key 1 lies in any subtree
        return np.array(
            [
                int(
                    tree_1.replace(" ", "")
                    .replace("(..)", "")
                    .replace("(,,)", "")
                    .lower()
                    in subtrees_0
                )
                for tree_1, subtrees_0 in zip(trees_1, all_subtrees_0)
            ]
        )


@prerequisites(AllenConstituencyParsingOp)
class FuzzyConstituencySubtreeSubpopulation(ScoreSubpopulation):
    def score(
        self,
        batch: DataPanel,
        columns: List[str],
        *args,
        **kwargs,
    ) -> np.ndarray:
        # Require that the number of keys is exactly 2
        assert len(columns) == 2, "Must specify exactly 2 keys."

        # Retrieve the trees
        trees = {
            col: lookup(batch, AllenConstituencyParsingOp, [col]) for col in columns
        }
        trees_0, trees_1 = trees[columns[0]], trees[columns[1]]

        # Convert the trees corresponding to key 0 to NLTK trees
        trees_0 = [nltk.Tree.fromstring(tree) for tree in trees_0]

        # Find all subtrees of these trees
        all_subtrees_0 = [
            set(
                [
                    str(t).replace("\n", "").replace(" ", "").lower()
                    for t in tree_0.subtrees()
                ]
            )
            for tree_0 in trees_0
        ]

        # Output a fuzzy score if the tree corresponding to key 1 is similar to any
        # subtree
        return np.array(
            [
                max(
                    [
                        fuzz.partial_ratio(
                            tree_1.replace(" ", "")
                            .replace("(..)", "")
                            .replace("(,,)", "")
                            .lower(),
                            subtree,
                        )
                        for subtree in subtrees_0
                    ]
                )
                for tree_1, subtrees_0 in zip(trees_1, all_subtrees_0)
            ]
        )
