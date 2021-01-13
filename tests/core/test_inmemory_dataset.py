from unittest import TestCase

from robustnessgym.core.dataformats.inmemory import InMemoryDataset


class TestInMemoryDataset(TestCase):
    def setUp(self):
        self.batch = {
            "a": [{"e": 1}, {"e": 2}, {"e": 3}, {"e": 4}],
            "b": ["u", "v", "w", "x"],
        }
        self.dataset = InMemoryDataset.from_batch(self.batch)

    def test_init(self):
        # Empty dataset
        dataset = InMemoryDataset()
        self.assertEqual(len(dataset), 0, "Must be zero length.")

        # Dataset from batch: illegal
        with self.assertRaises(AssertionError):
            InMemoryDataset.from_batch(
                {
                    "a": [1, 2, 3],
                    "b": [1, 2, 3, 4],
                }
            )

        dataset = InMemoryDataset.from_batch(
            {
                "a": [1, 2, 3, 4],
                "b": ["u", "v", "w", "x"],
            }
        )
        self.assertEqual(len(dataset), 4)
        self.assertEqual(set(dataset.column_names), {"a", "b"})

        dataset = InMemoryDataset.from_batch(
            {
                "a": [{"e": 1}, {"e": 2}, {"e": 3}, {"e": 4}],
                "b": ["u", "v", "w", "x"],
            }
        )
        self.assertEqual(len(dataset), 4)
        self.assertEqual(set(dataset.column_names), {"a", "b"})

    def test_getindex(self):
        self.assertEqual(
            self.dataset[0],
            {
                "a": {"e": 1},
                "b": "u",
            },
        )

        self.assertEqual(
            self.dataset[1:3],
            {
                "a": [{"e": 2}, {"e": 3}],
                "b": ["v", "w"],
            },
        )

        self.assertEqual(
            self.dataset[:-1],
            {
                "a": [{"e": 1}, {"e": 2}, {"e": 3}],
                "b": ["u", "v", "w"],
            },
        )

        self.assertEqual(
            self.dataset[::-1],
            {
                "a": [{"e": 4}, {"e": 3}, {"e": 2}, {"e": 1}],
                "b": ["x", "w", "v", "u"],
            },
        )

    def test_append_1(self):
        batch = {
            "a": [1, 2, 3, 4],
            "b": ["u", "v", "w", "x"],
        }
        self.dataset.append(batch)

        self.assertEqual(self.dataset[-1], {"a": 4, "b": "x"})
        self.assertEqual(len(self.dataset), 8)

    def test_append_2(self):
        batch = {
            "a": 1,
            "b": "u",
        }
        self.dataset.append(batch)

        self.assertEqual(self.dataset[-1], {"a": 1, "b": "u"})
        self.assertEqual(len(self.dataset), 5)

    def test_map_1(self):
        """Map, with_indices=False, batched=False."""
        dataset = self.dataset.map(
            lambda example: {"c": example["a"]},
            with_indices=False,
            input_columns=None,
            batched=False,
            batch_size=1000,
            drop_last_batch=False,
            remove_columns=None,
            required_columns=None,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset.column_names), {"a", "b", "c"})

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.batch,
        )

    def test_map_2(self):
        """Map, with_indices=True, batched=False."""
        dataset = self.dataset.map(
            lambda example, index: {"c": example["a"]},
            with_indices=True,
            input_columns=None,
            batched=False,
            batch_size=1000,
            drop_last_batch=False,
            remove_columns=None,
            required_columns=None,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset.column_names), {"a", "b", "c"})

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.batch,
        )

    def test_map_3(self):
        """Map, with_indices=False, batched=True."""
        dataset = self.dataset.map(
            lambda example: {"c": example["a"]},
            with_indices=False,
            input_columns=None,
            batched=True,
            batch_size=1000,
            drop_last_batch=False,
            remove_columns=None,
            required_columns=None,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset.column_names), {"a", "b", "c"})

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.batch,
        )

    def test_map_4(self):
        """Map, with_indices=True, batched=True."""
        dataset = self.dataset.map(
            lambda example, index: {"c": example["a"]},
            with_indices=True,
            input_columns=None,
            batched=True,
            batch_size=1000,
            drop_last_batch=False,
            remove_columns=None,
            required_columns=None,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset.column_names), {"a", "b", "c"})

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.batch,
        )

    def test_map_5(self):
        """Map, function=None, with_indices=False, batched=False."""
        dataset = self.dataset.map(
            None,
            with_indices=False,
            batched=False,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset.column_names), {"a", "b"})

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.batch,
        )

    def test_filter_1(self):
        dataset = self.dataset.filter(
            lambda example: example["a"] == {"e": 1},
            with_indices=False,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset.column_names), {"a", "b"})

        # Dataset has the right data
        self.assertEqual(
            dataset[:],
            {
                "a": [{"e": 1}],
                "b": ["u"],
            },
        )

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.batch,
        )

    def test_filter_2(self):
        dataset = self.dataset.filter(
            lambda example, index: example["a"] == {"e": 1},
            with_indices=True,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset.column_names), {"a", "b"})

        # Dataset has the right data
        self.assertEqual(
            dataset[:],
            {
                "a": [{"e": 1}],
                "b": ["u"],
            },
        )

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.batch,
        )
