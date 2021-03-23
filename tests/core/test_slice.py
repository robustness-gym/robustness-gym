"""Unittests for Slices."""
from unittest import TestCase

from robustnessgym.core.slice import Slice
from tests.testbeds import MockTestBedv0


class TestSlice(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_from_dataset(self):
        # Create a slice
        sl = Slice.from_dataset(self.testbed.dataset)
        # Compare the slice identifier
        self.assertEqual(str(sl), "RGSlice[num_rows: 6](MockDataset(version=1.0))")
        # Length of the slice
        self.assertEqual(len(sl), 6)
        # Lineage of the slice
        self.assertEqual(sl.lineage, [("Dataset", "MockDataset(version=1.0)")])
