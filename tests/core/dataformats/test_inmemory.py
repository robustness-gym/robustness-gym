"""Unittests for Datasets."""
import os
import shutil
from copy import deepcopy
from unittest import TestCase

import numpy as np

from robustnessgym.core.dataformats.inmemory import InMemoryDataset
from tests.testbeds import MockTestBedv0


class TestInMemoryDataset(TestCase):
    def setUp(self):
        # Arrange
        self.testbed = MockTestBedv0()

    def _build_inmemorydataset(self):
        # Build a dataset from a batch
        return InMemoryDataset.from_batch(batch=self.testbed.batch)

    def test_from_batch(self):
        # Build a dataset from a batch
        dataset = self._build_inmemorydataset()

        self.assertEqual(
            set(dataset.column_names),
            {"text", "label", "z", "fast", "metadata", "index"},
        )
        self.assertEqual(len(dataset), 6)

    def test_from_batches(self):
        # Build a dataset from a list of batches
        dataset = InMemoryDataset.from_batches(
            batches=[deepcopy(self.testbed.batch) for _ in range(2)]
        )

        self.assertEqual(
            set(dataset.column_names),
            {"text", "label", "z", "fast", "metadata", "index"},
        )
        self.assertEqual(len(dataset), 6 * 2)

    def test_indexing(self):
        # Build a dataset from a batch
        dataset = self._build_inmemorydataset()

        self.assertEqual(dataset[0], {k: v[0] for k, v in self.testbed.batch.items()})
        self.assertEqual(dataset[:1], {k: v[:1] for k, v in self.testbed.batch.items()})
        self.assertEqual(dataset[:], self.testbed.batch)
        self.assertEqual(dataset["text"], self.testbed.batch["text"])
        self.assertEqual(dataset[[0, 1]], dataset[:2])
        self.assertEqual(dataset[np.array([0, 1])], dataset[:2])

    def test_add_column(self):
        # Build a dataset from a batch
        dataset = self._build_inmemorydataset()

        # Add a new column to the dataset
        dataset.add_column("test", list(range(6)))
        self.assertEqual(dataset["test"], list(range(6)))

        with self.assertRaises(AssertionError):
            # Invalid addition of a column to the dataset
            dataset.add_column("invalid", list(range(7)))

    def test_save_load(self):
        # Build a dataset from a batch
        dataset = self._build_inmemorydataset()

        # Save to disk
        # Create a temporary directory
        os.makedirs("tmp", exist_ok=True)

        # Save the dataset to disk
        dataset.save_to_disk(path="tmp")

        # Load the dataset from disk
        loaded_dataset = InMemoryDataset.load_from_disk(path="tmp")

        # Remove the temporary directory
        shutil.rmtree("tmp")

        self.assertEqual(dataset.features, loaded_dataset.features)
