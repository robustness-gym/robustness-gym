from typing import List, Tuple, Dict

import fuzzywuzzy.fuzz as fuzz
import numpy as np
from nltk import Tree

from robustnessgym.cached_ops.allen.constituency_parser.constituency_parser import AllenConstituencyParser
from robustnessgym.identifier import Identifier
from robustnessgym.slicebuilders.subpopulations.score.score import HasScore


class HasConstituencyOverlap(HasScore,
                             AllenConstituencyParser,
                             ):

    def __init__(self,
                 intervals: List[Tuple[int, int]],
                 *args,
                 **kwargs
                 ):
        super(HasConstituencyOverlap, self).__init__(
            intervals=intervals,
            *args,
            **kwargs
        )

    def score(self,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> np.ndarray:
        # Require that the number of keys is exactly 2
        assert len(keys) == 2, "Must specify exactly 2 keys."

        # Retrieve the trees
        trees = AllenConstituencyParser.retrieve(batch=batch, columns=[[key] for key in keys])
        trees_0, trees_1 = trees[keys[0]], trees[keys[1]]

        # Fuzzy match the trees and return the scores
        return np.array([
            fuzz.partial_token_set_ratio(
                tree_0.replace("(", "").replace(")", "").replace(" ", ""),
                tree_1.replace("(", "").replace(")", "").replace(" ", "")
            )
            for tree_0, tree_1 in zip(trees_0, trees_1)
        ])


class HasConstituencySubtree(HasScore,
                             AllenConstituencyParser,
                             ):

    def __init__(self,
                 *args,
                 **kwargs):
        super(HasConstituencySubtree, self).__init__(
            intervals=[(1, 1)],
            identifiers=[
                Identifier(name=self.__class__.__name__)
            ],
            *args,
            **kwargs
        )

    def score(self,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> np.ndarray:
        # Require that the number of keys is exactly 2
        assert len(keys) == 2, "Must specify exactly 2 keys."

        # Retrieve the trees
        trees = AllenConstituencyParser.retrieve(batch=batch, columns=[[key] for key in keys])
        trees_0, trees_1 = trees[keys[0]], trees[keys[1]]

        # Convert the trees corresponding to key 0 to NLTK trees
        trees_0 = [Tree.fromstring(tree) for tree in trees_0]

        # Find all subtrees of these trees
        all_subtrees_0 = [
            set([str(t).replace("\n", "").replace(" ", "").lower() for t in tree_0.subtrees()])
            for tree_0 in trees_0
        ]

        # Output a score of 1 if the tree corresponding to key 1 lies in any subtree
        return np.array([
            int(tree_1.replace(" ", "").replace("(..)", "").replace("(,,)", "").lower() in subtrees_0)
            for tree_1, subtrees_0 in zip(trees_1, all_subtrees_0)
        ])


class HasFuzzyConstituencySubtree(HasScore,
                                  AllenConstituencyParser,
                                  ):

    def __init__(self,
                 intervals: List[Tuple[int, int]],
                 *args,
                 **kwargs):
        super(HasFuzzyConstituencySubtree, self).__init__(
            intervals=intervals,
            *args,
            **kwargs
        )

    def score(self,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> np.ndarray:
        # Require that the number of keys is exactly 2
        assert len(keys) == 2, "Must specify exactly 2 keys."

        # Retrieve the trees
        trees = AllenConstituencyParser.retrieve(batch=batch, columns=[[key] for key in keys])
        trees_0, trees_1 = trees[keys[0]], trees[keys[1]]

        # Convert the trees corresponding to key 0 to NLTK trees
        trees_0 = [Tree.fromstring(tree) for tree in trees_0]

        # Find all subtrees of these trees
        all_subtrees_0 = [
            set([str(t).replace("\n", "").replace(" ", "").lower() for t in tree_0.subtrees()])
            for tree_0 in trees_0
        ]

        # Output a fuzzy score if the tree corresponding to key 1 is similar to any subtree
        return np.array([
            max([fuzz.partial_ratio(tree_1.replace(" ", "").replace("(..)", "").replace("(,,)", "").lower(), subtree)
                 for subtree in subtrees_0])
            for tree_1, subtrees_0 in zip(trees_1, all_subtrees_0)
        ])
