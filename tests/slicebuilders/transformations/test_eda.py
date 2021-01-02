"""Unittests for the EasyDataAugmentation class."""
import random
from unittest import TestCase

import numpy as np

from robustnessgym.slicebuilders.transformations.eda import EasyDataAugmentation
from tests.testbeds import MockTestBedv0


class TestEasyDataAugmentation(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_apply(self):
        # Create the EDA SliceBuilder
        eda = EasyDataAugmentation(num_transformed=3)

        # Set the seed
        random.seed(0)
        np.random.seed(0)

        for i, identifier in enumerate(eda.identifiers):
            self.assertEqual(
                str(identifier),
                f"EasyDataAugmentation-{i + 1}(alpha_sr=0.1, alpha_ri=0.1, "
                f"alpha_rs=0.1, p_rd=0.1)",
            )

        # Apply it
        dataset, slices, slice_membership = eda(self.testbed.dataset, columns=["text"])

        # All the sizes match up
        self.assertEqual(len(dataset), len(self.testbed.dataset))
        for sl in slices:
            self.assertEqual(len(sl), len(self.testbed.dataset))
        self.assertEqual(slice_membership.shape, (6, 3))

        # Everything was transformed
        self.assertTrue(np.all(slice_membership))

        # Checking that the transformed text matches
        self.assertEqual(
            slices[0]["text"],
            [
                "the man is military man walking",
                "the constitute man is running",
                "the woman is sprinting",
                "the woman is",
                "the hobbit is flying",
                "the hobbit is swimming",
            ],
        )

        # Dataset interaction history updated correctly
        self.assertEqual(
            len(dataset.fetch_tape(["slicebuilders", "transformation"]).history), 3
        )
