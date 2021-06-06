from unittest import TestCase

import numpy as np

from robustnessgym.ops.allen.constituency_parser import AllenConstituencyParser
from robustnessgym.slicebuilders.subpopulations.constituency_overlap import (
    ConstituencyOverlapSubpopulation,
    ConstituencySubtreeSubpopulation,
    FuzzyConstituencySubtreeSubpopulation,
)
from tests.testbeds import MockTestBedv1


class TestConstituencyOverlap(TestCase):
    testbed = MockTestBedv1()
    acp = AllenConstituencyParser()
    testbed.dataset = acp(testbed.dataset, columns=["text_a"])
    testbed.dataset = acp(testbed.dataset, columns=["text_b"])

    def test_has_constituency_overlap(self):
        # Create the constituency overlap subpopulation
        cos = ConstituencyOverlapSubpopulation(
            intervals=[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        )
        print(self.testbed.dataset[:])
        print(len(self.testbed.dataset), self.testbed.dataset)
        print(cos.score(self.testbed.dataset[:], columns=["text_a", "text_b"]))
        self.assertTrue(
            np.allclose(
                cos.score(self.testbed.dataset[:], columns=["text_a", "text_b"]),
                [100] * 4,
            )
        )

    def test_has_constituency_subtree(self):
        # Create the constituency subtree subpopulation
        css = ConstituencySubtreeSubpopulation()
        self.assertTrue(
            np.allclose(
                css.score(self.testbed.dataset[:], columns=["text_a", "text_b"]),
                [1] * 4,
            )
        )

    def test_has_fuzzy_constituency_subtree(self):
        # Create the fuzzy constituency subtree subpopulation
        fcss = FuzzyConstituencySubtreeSubpopulation(
            intervals=[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        )
        self.assertTrue(
            np.allclose(
                fcss.score(self.testbed.dataset[:], columns=["text_a", "text_b"]),
                [100] * 4,
            )
        )
