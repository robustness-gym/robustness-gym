"""
Unittests for CachedOperations.
"""

from unittest import TestCase

from robustnessgym.core.cachedops import CachedOperation
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import strings_as_json
from tests.testbeds import MockTestBedv0


def a_single_column_apply_fn(batch, columns):
    assert len(columns) == 1 and type(batch[columns[0]][0]) == int
    return [e * 7 + 3.14 for e in batch[columns[0]]]


def a_multi_column_apply_fn(batch, columns):
    assert len(columns) == 2
    return [e[0] * 0.1 + e[1] * 0.3 for e in zip(batch[columns[0]], batch[columns[1]])]


class TestCachedOperation(TestCase):

    def setUp(self):
        # Arrange
        self.cachedop = CachedOperation(
            apply_fn=a_single_column_apply_fn,
            identifier=Identifier(_name='TestCachedOperation'),
        )

        self.testbed = MockTestBedv0()

        self.multicol_cachedop = CachedOperation(
            apply_fn=a_multi_column_apply_fn,
            identifier=Identifier(_name='TestCachedOperation', columns='multiple')
        )

    def test_repr(self):
        self.assertEqual(str(self.cachedop), 'TestCachedOperation')

    def test_endtoend(self):
        # Apply to the dataset
        self.cachedop(self.testbed.dataset, columns=['label'])

        # Check that the dataset remains the same
        self.assertEqual(self.testbed.dataset.features, self.testbed.original_dataset.features)

        # Apply and store
        self.testbed.dataset = self.cachedop(self.testbed.dataset, columns=['label'])

        # The dataset should have changed
        self.assertNotEqual(self.testbed.dataset.features, self.testbed.original_dataset.features)

        # It should contain the special cache key
        self.assertTrue('cache' in self.testbed.dataset.features)

        # The interaction tape should contain the history of this operation
        self.assertTrue(self.testbed.dataset.fetch_tape(path=['cachedoperations']).history ==
                        {(self.cachedop.identifier, 'label'): 0})

        # Retrieve the information that was stored using the instance
        self.assertEqual(self.cachedop.retrieve(self.testbed.dataset[:], columns=['label']),
                         {'label': [3.14, 3.14, 10.14, 10.14, 3.14, 3.14]})

        # Retrieve the information that was stored using the CachedOperation class, and an identifier
        self.assertEqual(CachedOperation.retrieve(self.testbed.dataset[:], columns=['label'],
                                                  identifier=self.cachedop.identifier),
                         {'label': [3.14, 3.14, 10.14, 10.14, 3.14, 3.14]})

        # Retrieve the information that was stored using the CachedOperation class: fails without the identifier
        with self.assertRaises(ValueError):
            CachedOperation.retrieve(self.testbed.dataset[:], columns=['label'])

        # Retrieve the information that was stored, and process it with a function
        self.assertEqual(self.cachedop.retrieve(self.testbed.dataset[:], columns=['label'],
                                                proc_fns=lambda decoded_batch: [x + 0.01 for x in decoded_batch]),
                         {'label': [3.15, 3.15, 10.15, 10.15, 3.15, 3.15]})

    def test_multiple_calls(self):
        # Apply to multiple columns of the dataset directly: fails since the function requires single column
        with self.assertRaises(AssertionError):
            self.cachedop(self.testbed.dataset, columns=['label', 'fast'])

        # Create an additional integer column in the dataset
        dataset = self.testbed.dataset.map(lambda x: {'otherlabel': x['label'] + 1})

        # Apply to multiple columns of the dataset in sequence
        dataset_0_0 = self.cachedop(dataset, columns=['label'])
        dataset_0_1 = self.cachedop(dataset_0_0, columns=['z'])

        # Check that the cache is populated with the processed columns
        self.assertTrue('label' in dataset_0_0.features['cache'][str(self.cachedop.identifier)] and
                        'z' not in dataset_0_0.features['cache'][str(self.cachedop.identifier)])
        self.assertTrue('label' in dataset_0_1.features['cache'][str(self.cachedop.identifier)] and
                        'z' in dataset_0_1.features['cache'][str(self.cachedop.identifier)])

        # Apply to multiple columns of the dataset, in reverse order
        dataset_1_0 = self.cachedop(dataset, columns=['z'])
        dataset_1_1 = self.cachedop(dataset_1_0, columns=['label'])

        # Check that the cache is populated with the processed columns
        self.assertTrue('label' not in dataset_1_0.features['cache'][str(self.cachedop.identifier)] and
                        'z' in dataset_1_0.features['cache'][str(self.cachedop.identifier)])
        self.assertTrue('label' in dataset_1_1.features['cache'][str(self.cachedop.identifier)] and
                        'z' in dataset_1_1.features['cache'][str(self.cachedop.identifier)])

        # Retrieving information fails if the columns are passed together in a single list
        with self.assertRaises(KeyError) as context:
            self.cachedop.retrieve(dataset_1_1[:], columns=['label', 'z'])
        print("Fails:", str(context.exception))

        # Retrieving information succeeds when the columns are passed separately
        self.assertTrue(
            self.cachedop.retrieve(dataset_1_1[:], columns=[['label'], ['z']]),
            {'label': [3.14, 3.14, 10.14, 10.14, 3.14, 3.14],
             'z': [10.14, 3.14, 10.14, 3.14, 10.14, 3.14]}
        )

    def test_multicolumn(self):
        # Apply the multi-column cached operation
        dataset = self.multicol_cachedop(self.testbed.dataset, columns=['label', 'z'])

        # Check that caching happens and that the cached values are correct
        self.assertTrue(strings_as_json(['label', 'z']) in
                        dataset.features['cache'][str(self.multicol_cachedop.identifier)])
        self.assertEqual(self.multicol_cachedop.retrieve(dataset[:], columns=['label', 'z']),
                         {'["label", "z"]': [0.3, 0.0, 0.4, 0.1, 0.3, 0.0]})

        # Apply the single-column cached operation
        dataset = self.cachedop(dataset, columns=['label'])
        dataset = self.cachedop(dataset, columns=['z'])

        # Now recheck that everything can be retrieved correctly
        self.assertTrue(strings_as_json(['label', 'z']) in
                        dataset.features['cache'][str(self.multicol_cachedop.identifier)])
        self.assertEqual(self.multicol_cachedop.retrieve(dataset[:], columns=['label', 'z']),
                         {'["label", "z"]': [0.3, 0.0, 0.4, 0.1, 0.3, 0.0]})
        self.assertEqual(self.cachedop.retrieve(dataset[:], columns=['label']),
                         {'label': [3.14, 3.14, 10.14, 10.14, 3.14, 3.14]})
        self.assertEqual(self.cachedop.retrieve(dataset[:], columns=['z']),
                         {'z': [10.14, 3.14, 10.14, 3.14, 10.14, 3.14]})
