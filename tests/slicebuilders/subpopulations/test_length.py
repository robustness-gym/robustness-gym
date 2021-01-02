from unittest import TestCase

import numpy as np

from robustnessgym.cachedops.spacy import Spacy
from robustnessgym.slicebuilders.subpopulations.length import LengthSubpopulation
from tests.testbeds import MockTestBedv0


class TestLengthSubpopulation(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()
        self.testbed.dataset = Spacy()(self.testbed.dataset, columns=["text"])

    def test_score(self):
        # Create the length subpopulation
        length = LengthSubpopulation(intervals=[(1, 3), (4, 5)])

        # Compute scores
        scores = length.score(self.testbed.dataset[:], columns=["text"])
        self.assertTrue(np.allclose(scores, np.array([5, 5, 5, 5, 5, 5])))

        # Apply the subpopulation
        dataset, slices, slice_matrix = length(self.testbed.dataset, columns=["text"])

        # Check that the slice membership lines up
        self.assertTrue(np.allclose(slice_matrix, np.array([[0, 1]] * 6)))
