"""
Unittests for Datasets.
"""
import os
import shutil
from unittest import TestCase

import jsonlines
from robustnessgym.core.dataset import Dataset, transpose_batch
from robustnessgym.core.identifier import Identifier
from tests.testbeds import MockTestBedv0


class TestDataset(TestCase):

    def setUp(self):
        # Arrange
        self.testbed = MockTestBedv0()

    def test_from_batch(self):
        # Build a dataset from a batch
        dataset = Dataset.from_batch({
            'a': [1, 2, 3],
            'b': [True, False, True],
            'c': ['x', 'y', 'z'],
            'd': [{'e': 2}, {'e': 3}, {'e': 4}]
        },
            identifier=Identifier(_name='MyDataset')
        )

        self.assertEqual(set(dataset.column_names), {'a', 'b', 'c', 'd', 'index'})
        self.assertEqual(len(dataset), 3)

    def test_from_batches(self):
        # Build a dataset from multiple batches
        dataset = Dataset.from_batches([{
            'a': [1, 2, 3],
            'b': [True, False, True],
            'c': ['x', 'y', 'z'],
            'd': [{'e': 2}, {'e': 3}, {'e': 4}]
        }] * 3,
                                       identifier=Identifier(_name='MyDataset')
                                       )

        self.assertEqual(set(dataset.column_names), {'a', 'b', 'c', 'd', 'index'})
        self.assertEqual(len(dataset), 9)

    def test_from_json(self):
        # Create a temporary directory
        os.mkdir('tmp')

        # Create a json file with data
        with jsonlines.open('tmp/data.jsonl', 'w') as writer:
            writer.write_all(
                transpose_batch({
                    'a': [1, 2, 3],
                    'b': [True, False, True],
                    'c': ['x', 'y', 'z'],
                    'd': [{'e': 2}, {'e': 3}, {'e': 4}]
                })
            )

        # Load the dataset
        dataset = Dataset.from_json(
            json_path='tmp/data.jsonl',
            identifier=Identifier(_name='MockJSONDataset'),
        )

        self.assertEqual(set(dataset.column_names), {'a', 'b', 'c', 'd', 'index'})
        self.assertEqual(len(dataset), 3)

        # Remove the temporary directory
        shutil.rmtree('tmp')

    def test_save_load(self):
        # Create a temporary directory
        os.mkdir('tmp')

        # Save the dataset to disk
        self.testbed.dataset.save(path='tmp')

        # Load the dataset from disk
        dataset = Dataset.load(path='tmp')

        # Remove the temporary directory
        shutil.rmtree('tmp')

        self.assertEqual(dataset.features, self.testbed.dataset.features)

    def test_map(self):
        # Map over the dataset
        dataset = self.testbed.dataset.map(lambda x: {'otherlabel': x['label'] + 1})
        self.assertTrue('otherlabel' in dataset.column_names)
        self.assertEqual(dataset['otherlabel'], [1, 1, 2, 2, 1, 1])

    def test_batch(self):
        # Check that we can make batches of different sizes
        self.assertEqual(len(list(self.testbed.dataset.batch(4))), 2)
        self.assertEqual(len(list(self.testbed.dataset.batch(3))), 2)
        self.assertEqual(len(list(self.testbed.dataset.batch(2))), 3)
        self.assertEqual(len(list(self.testbed.dataset.batch(1))), 6)

        # Check that batches of 2 are correct
        self.assertEqual(list(self.testbed.dataset.batch(2)),
                         [{'fast': [False, True], 'index': ['0', '1'], 'label': [0, 0],
                           'metadata': [{'source': 'real'}, {'source': 'real'}],
                           'text': ['The man is walking.', 'The man is running.'], 'z': [1, 0]},
                          {'fast': [True, False], 'index': ['2', '3'], 'label': [1, 1],
                           'metadata': [{'source': 'real'}, {'source': 'real'}],
                           'text': ['The woman is sprinting.', 'The woman is resting.'], 'z': [1, 0]},
                          {'fast': [False, False], 'index': ['4', '5'], 'label': [0, 0],
                           'metadata': [{'source': 'fictional'}, {'source': 'fictional'}],
                           'text': ['The hobbit is flying.', 'The hobbit is swimming.'], 'z': [1, 0]}])

    def test_chain(self):
        # Chain the dataset with itself
        dataset = Dataset.chain(
            [self.testbed.dataset, self.testbed.dataset],
            identifier=Identifier(_name='MockChainedDataset')
        )

        # Check that the elements match up
        for i, x in enumerate(dataset):
            self.assertEqual(x, self.testbed.dataset[i % len(self.testbed.dataset)])

        self.assertEqual(len(dataset), len(self.testbed.dataset) * 2)

    def test_interleave(self):
        # Interleave the dataset with itself
        dataset = Dataset.interleave(
            [self.testbed.dataset, self.testbed.dataset],
            identifier=Identifier(_name='MockInterleavedDataset')
        )

        # Check that the elements match up
        for i, x in enumerate(dataset):
            self.assertEqual(x, self.testbed.dataset[i // 2])

        self.assertEqual(len(dataset), len(self.testbed.dataset) * 2)

    def test_load_dataset(self):
        # Load the first 20 examples of the boolq dataset
        dataset = Dataset.load_dataset('boolq', split='train[:20]')

        # Check that we got 20 examples
        self.assertTrue(isinstance(dataset, Dataset))
        self.assertEqual(len(dataset), 20)
