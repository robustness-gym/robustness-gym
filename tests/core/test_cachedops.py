"""Unittests for Operations."""

from unittest import TestCase

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.operation import Operation
from tests.testbeds import MockTestBedv0


def a_single_column_apply_fn(batch, columns):
    """Simple function that applies to a single column."""
    assert len(columns) == 1 and type(batch[columns[0]][0]) == int
    return [e * 7 + 3.14 for e in batch[columns[0]]]


def a_multi_column_apply_fn(batch, columns):
    """Simple function that applies to multiple columns."""
    assert len(columns) == 2
    return [e[0] * 0.1 + e[1] * 0.3 for e in zip(batch[columns[0]], batch[columns[1]])]


class TestOperation(TestCase):
    def setUp(self):
        self.op = Operation(
            apply_fn=a_single_column_apply_fn,
            identifier=Identifier(_name="TestOperation"),
        )

        self.testbed = MockTestBedv0()

        self.multicol_op = Operation(
            apply_fn=a_multi_column_apply_fn,
            identifier=Identifier(_name="TestOperation", to="multiple"),
        )

    def test_repr(self):
        self.assertEqual(str(self.op), "TestOperation")

    def test_endtoend(self):
        # Apply to the dataset
        self.op(self.testbed.dataset, columns=["label"])

        # Check that the dataset remains the same
        self.assertEqual(
            self.testbed.dataset.features, self.testbed.original_dataset.features
        )

        # Apply and store
        self.testbed.dataset = self.op(self.testbed.dataset, columns=["label"])

        # The dataset should have changed
        self.assertNotEqual(
            self.testbed.dataset.features, self.testbed.original_dataset.features
        )

        # The interaction tape should contain the history of this operation
        # self.assertTrue(
        #     self.testbed.dataset.fetch_tape(path=["cachedoperations"]).history
        #     == {(self.op.identifier, "label"): 0}
        # )

        # Retrieve the information that was stored using the instance
        self.assertEqual(
            self.op.retrieve(self.testbed.dataset[:], columns=["label"]),
            [3.14, 3.14, 10.14, 10.14, 3.14, 3.14],
        )

        # Retrieve the information that was stored using the Operation class,
        # and an identifier
        self.assertEqual(
            Operation.retrieve(
                self.testbed.dataset[:],
                columns=["label"],
                identifier=self.op.identifier,
            ),
            [3.14, 3.14, 10.14, 10.14, 3.14, 3.14],
        )

        # Retrieve the information that was stored using the Operation class:
        # fails without the identifier
        with self.assertRaises(ValueError):
            Operation.retrieve(self.testbed.dataset[:], columns=["label"])

        # Retrieve the information that was stored, and process it with a function
        self.assertEqual(
            self.op.retrieve(
                self.testbed.dataset[:],
                columns=["label"],
                proc_fns=lambda decoded_batch: [x + 0.01 for x in decoded_batch],
            ),
            [3.15, 3.15, 10.15, 10.15, 3.15, 3.15],
        )

    def test_multiple_calls(self):
        # Apply to multiple columns of the dataset directly: fails since the function
        # requires single column
        with self.assertRaises(AssertionError):
            self.op(self.testbed.dataset, columns=["label", "fast"])

        # Create an additional integer column in the dataset
        dataset = self.testbed.dataset.update(lambda x: {"otherlabel": x["label"] + 1})

        # Apply to multiple columns of the dataset in sequence
        dataset_0_0 = self.op(dataset, columns=["label"])
        dataset_0_1 = self.op(dataset_0_0, columns=["z"])

        # Check that the additional columns were added
        self.assertEqual(len(dataset.column_names) + 1, len(dataset_0_0.column_names))
        self.assertEqual(
            len(dataset_0_0.column_names) + 1, len(dataset_0_1.column_names)
        )

        # Apply to multiple columns of the dataset, in reverse order
        dataset_1_0 = self.op(dataset, columns=["z"])
        dataset_1_1 = self.op(dataset_1_0, columns=["label"])

        # Check that the cache is populated with the processed columns
        self.assertEqual(len(dataset.column_names) + 1, len(dataset_1_0.column_names))
        self.assertEqual(
            len(dataset_1_0.column_names) + 1, len(dataset_1_1.column_names)
        )

        # Retrieving information fails if the columns are passed together in a single
        # list
        with self.assertRaises(KeyError) as context:
            self.op.retrieve(dataset_1_1[:], columns=["label", "z"])
        print("Fails:", str(context.exception))

        # Retrieving information succeeds when the columns are passed separately
        self.assertTrue(
            self.op.retrieve(dataset_1_1[:], columns=[["label"], ["z"]]),
            {
                "label": [3.14, 3.14, 10.14, 10.14, 3.14, 3.14],
                "z": [10.14, 3.14, 10.14, 3.14, 10.14, 3.14],
            },
        )

    def test_multicolumn(self):
        # Apply the multi-column operation
        dataset = self.multicol_op(self.testbed.dataset, columns=["label", "z"])

        # Check that caching happens and that the cached values are correct
        self.assertEqual(
            self.multicol_op.retrieve(dataset[:], columns=["label", "z"]),
            {("label", "z"): [0.3, 0.0, 0.4, 0.1, 0.3, 0.0]},
        )

        # Apply the single-column operation
        dataset = self.op(dataset, columns=["label"])
        dataset = self.op(dataset, columns=["z"])

        # Now recheck that everything can be retrieved correctly
        self.assertEqual(
            self.multicol_op.retrieve(dataset[:], columns=["label", "z"]),
            {("label", "z"): [0.3, 0.0, 0.4, 0.1, 0.3, 0.0]},
        )
        self.assertEqual(
            self.op.retrieve(dataset[:], columns=["label"]),
            [3.14, 3.14, 10.14, 10.14, 3.14, 3.14],
        )
        self.assertEqual(
            self.op.retrieve(dataset[:], columns=["z"]),
            [10.14, 3.14, 10.14, 3.14, 10.14, 3.14],
        )
