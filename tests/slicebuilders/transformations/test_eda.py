import numpy as np
from robustnessgym import *
from unittest import TestCase
from tests.testbeds import MockTestBedv0


class TestEasyDataAugmentation(TestCase):

    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_apply(self):
        # Set the seed
        np.random.seed(0)

        # Create the EDA SliceBuilder
        eda = EasyDataAugmentation(num_transformed=3)

        # Apply it
        dataset, slices, slice_membership = eda(self.testbed.dataset, columns=['text'])

        # All the sizes match up
        self.assertEqual(len(dataset), len(self.testbed.dataset))
        for sl in slices:
            self.assertEqual(len(sl), len(self.testbed.dataset))
        self.assertEqual(slice_membership.shape, (6, 3))

        # Everything was transformed
        self.assertTrue(np.all(slice_membership))

        # Checking that the transformed text matches
        self.assertEqual(slices[0]['text'],
                         ['the walk to man is walking', 'the man is exist running', 'the woman is sprint sprinting',
                          'the woman rest is resting', 'the hobbit pilot is flying', 'the hobbit is swimming'])

        # Dataset interaction history updated correctly
        self.assertEqual(len(dataset.fetch_tape(['slicebuilders', 'transformation']).history), 3)
