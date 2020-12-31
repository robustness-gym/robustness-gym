"""
Unittests for the NlpAugTransformation class.
"""
from unittest import TestCase

import numpy as np
from nlpaug.augmenter.word import SynonymAug
from nlpaug.flow import Sequential
from robustnessgym.slicebuilders.transformations.nlpaug.nlpaugtransformation import NlpAugTransformation
from tests.testbeds import MockTestBedv0


class TestNlpAugTransformation(TestCase):

    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_apply(self):
        # Set the seed
        np.random.seed(0)

        # Create the nlpaug transformation
        nlpaug_transformation = NlpAugTransformation(
            pipeline=Sequential(flow=[SynonymAug()]),
            num_transformed=3,
        )

        for i, identifier in enumerate(nlpaug_transformation.identifiers):
            self.assertEqual(str(identifier),
                             f'NlpAugTransformation-{i + 1}(pipeline=[Synonym_Aug(src=wordnet, '
                             f'action=substitute, method=word)])')

        # Apply it
        dataset, slices, slice_membership = nlpaug_transformation(self.testbed.dataset, columns=['text'])

        # All the sizes match up
        self.assertEqual(len(dataset), len(self.testbed.dataset))
        for sl in slices:
            self.assertEqual(len(sl), len(self.testbed.dataset))
        self.assertEqual(slice_membership.shape, (6, 3))

        # Everything was transformed
        self.assertTrue(np.all(slice_membership))

        # Dataset interaction history updated correctly
        self.assertEqual(len(dataset.fetch_tape(['slicebuilders', 'transformation']).history), 3)

        # Checking that the transformed text matches
        self.assertEqual(slices[0]['text'],
                         ['The man be walking.', 'The man be running.', 'The woman is sprint.',
                          'The woman is repose.', 'The hobbit be flying.', 'The hobbit is swimming.'])
