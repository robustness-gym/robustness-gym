"""Unittests for the FairseqBacktranslation class."""
import unittest
from unittest import TestCase

import numpy as np
import torch

from robustnessgym.slicebuilders.transformations.fairseq import FairseqBacktranslation
from tests.testbeds import MockTestBedv0


@unittest.skip("Downloads fairseq models during CI, which is slow.")
class TestFairseqBacktranslation(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_apply(self):
        # Set the seed
        np.random.seed(0)
        torch.random.manual_seed(0)

        # Create the backtranslation transformation
        self.backtranslation = FairseqBacktranslation(
            n_src2tgt=2,
            n_tgt2src=2,
            device="cpu",
        )

        # Apply it
        dataset, slices, slice_membership = self.backtranslation(
            self.testbed.dataset, columns=["text"]
        )

        # Checking that the transformed text matches
        self.assertEqual(
            slices[0]["text"],
            [
                "The man leaves.",
                "The man runs.",
                "The woman sprints.",
                "The Lady rests.",
                "The Hobbit is flying.",
                "The Hobbit floats.",
            ],
        )
