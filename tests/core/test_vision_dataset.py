from unittest import TestCase

import torch

from robustnessgym.core.dataformats.vision import VisionDataset
from tests.testbeds import MockVisionTestBed


class TestVisionDataset(TestCase):
    def setUp(self):
        self.testbed = MockVisionTestBed()
        self.dataset = self.testbed.dataset

    def test_to_dataloader(self):
        # test that correct objects are returned by dataloader
        for i, b in self.dataset.to_dataloader(columns=["i", "b"]):
            self.assertEqual(b[0], "u")
            self.assertTrue(torch.equal(i.squeeze(), self.testbed.images[0].load()))
            break

        # test that you can pass dataloader kwargs
        imgs = [img for img in self.dataset.to_dataloader(columns=["i"], batch_size=2)]
        self.assertEqual(len(imgs), 2)

    def test_to_dataloader_transforms(self):
        imgs = [
            img
            for img, _ in self.dataset.to_dataloader(
                columns=["i", "b"],
                column_to_transform={"i": lambda x: torch.zeros_like(x)},
            )
        ]
        self.assertTrue(torch.all(torch.eq(imgs[0], 0)))

    def test_init(self):
        # Empty dataset
        dataset = VisionDataset()
        self.assertEqual(len(dataset), 0, "Must be zero length.")

        # Dataset from batch: illegal
        with self.assertRaises(AssertionError):
            VisionDataset(
                {"a": [1, 2, 3], "b": [1, 2, 3, 4], "i": self.testbed.image_paths},
                img_columns="i",
            )

        dataset = VisionDataset(
            {
                "a": [1, 2, 3, 4],
                "b": ["u", "v", "w", "x"],
                "i": self.testbed.image_paths,
            },
            img_columns="i",
        )
        self.assertEqual(len(dataset), 4)
        self.assertEqual(set(dataset.column_names), {"a", "b", "i"})

        dataset = VisionDataset(
            {
                "a": [{"e": 1}, {"e": 2}, {"e": 3}, {"e": 4}],
                "b": ["u", "v", "w", "x"],
                "i": self.testbed.image_paths,
            },
            img_columns="i",
        )
        self.assertEqual(len(dataset), 4)
        self.assertEqual(set(dataset.column_names), {"a", "b", "i"})

    def test_getindex(self):
        self.assertEqual(
            self.dataset[0],
            {"a": {"e": 1}, "b": "u", "i": self.testbed.images[0]},
        )

        self.assertEqual(
            self.dataset[1:3],
            {"a": [{"e": 2}, {"e": 3}], "b": ["v", "w"], "i": self.testbed.images[1:3]},
        )

        self.assertEqual(
            self.dataset[:-1],
            {
                "a": [{"e": 1}, {"e": 2}, {"e": 3}],
                "b": ["u", "v", "w"],
                "i": self.testbed.images[:-1],
            },
        )

        self.assertEqual(
            self.dataset[::-1],
            {
                "a": [{"e": 4}, {"e": 3}, {"e": 2}, {"e": 1}],
                "b": ["x", "w", "v", "u"],
                "i": self.testbed.images[::-1],
            },
        )

    def test_append_1(self):
        batch = {
            "a": [1, 2, 3, 4],
            "b": ["u", "v", "w", "x"],
            "i": self.testbed.image_paths[::-1],
        }
        self.dataset.append(batch)

        self.assertEqual(
            self.dataset[-1], {"a": 4, "b": "x", "i": self.testbed.images[0]}
        )
        self.assertEqual(len(self.dataset), 8)

    def test_append_2(self):
        batch = {"a": 1, "b": "u", "i": self.testbed.image_paths[0]}
        self.dataset.append(batch)

        self.assertEqual(
            self.dataset[-1], {"a": 1, "b": "u", "i": self.testbed.images[0]}
        )
        self.assertEqual(len(self.dataset), 5)

    def test_map_1(self):
        """Map, with_indices=False, batched=False."""
        output = self.dataset.map(
            lambda example: {"c": example["a"], "k": example["i"]},
            with_indices=False,
            input_columns=None,
            batched=False,
            batch_size=1000,
            drop_last_batch=False,
        )

        # Original dataset is still the same
        self.assertEqual(self.dataset[:], self.testbed.batch)

        # Output is correct
        self.assertEqual(output["c"], self.dataset["a"])

        for i, img in enumerate(output["k"]):
            self.assertTrue(torch.equal(1.0 * img, self.testbed.image_tensors[i]))

    def test_map_2(self):
        """Map, with_indices=True, batched=False."""
        output = self.dataset.map(
            lambda example, index: {"c": example["a"], "k": example["i"]},
            with_indices=True,
            input_columns=None,
            batched=False,
            batch_size=1000,
            drop_last_batch=False,
        )

        # Original dataset is still the same
        self.assertEqual(self.dataset[:], self.testbed.batch)

        # Output is correct
        self.assertEqual(output["c"], self.dataset["a"])

        for i, img in enumerate(output["k"]):
            self.assertTrue(torch.equal(1.0 * img, self.testbed.image_tensors[i]))

    def test_map_3(self):
        """Map, with_indices=False, batched=True."""
        output = self.dataset.map(
            lambda example: {"c": example["a"], "k": example["i"]},
            with_indices=False,
            input_columns=None,
            batched=True,
            batch_size=1000,
            drop_last_batch=False,
        )

        # Original dataset is still the same
        self.assertEqual(self.dataset[:], self.testbed.batch)

        # Output is correct
        self.assertEqual(output["c"], self.dataset["a"])

        for i, img in enumerate(output["k"]):
            self.assertTrue(torch.equal(1.0 * img, self.testbed.image_tensors[i]))

    def test_map_4(self):
        """Map, with_indices=True, batched=True."""
        output = self.dataset.map(
            lambda example, index: {"c": example["a"], "k": example["i"]},
            with_indices=True,
            input_columns=None,
            batched=True,
            batch_size=1000,
            drop_last_batch=False,
        )

        # Original dataset is still the same
        self.assertEqual(self.dataset[:], self.testbed.batch)

        # Output is correct
        self.assertEqual(output["c"], self.dataset["a"])

        for i, img in enumerate(output["k"]):
            self.assertTrue(torch.equal(1.0 * img, self.testbed.image_tensors[i]))

    def test_map_5(self):
        """Map, function=None, with_indices=False, batched=False."""
        output = self.dataset.map(
            None,
            with_indices=False,
            batched=False,
        )

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.testbed.batch,
        )

        # Output is None
        self.assertEqual(output, None)

    def test_filter_1(self):
        dataset = self.dataset.filter(
            lambda example: example["a"] == {"e": 1}
            and torch.equal(1.0 * example["i"], self.testbed.image_tensors[0]),
            with_indices=False,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset.column_names), {"a", "b", "i"})

        # Dataset has the right data
        self.assertEqual(
            dataset["a"],
            [{"e": 1}],
        )

        self.assertEqual(
            dataset["b"],
            ["u"],
        )

        self.assertEqual(dataset["i"], [self.testbed.images[0]])

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.testbed.batch,
        )

    def test_filter_2(self):
        dataset = self.dataset.filter(
            lambda example, index: example["a"] == {"e": 1}
            and torch.equal(1.0 * example["i"], self.testbed.image_tensors[0]),
            with_indices=True,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset.column_names), {"a", "b", "i"})

        # Dataset has the right data
        self.assertEqual(
            dataset["a"],
            [{"e": 1}],
        )

        self.assertEqual(
            dataset["b"],
            ["u"],
        )

        self.assertEqual(dataset["i"], [self.testbed.images[0]])

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.testbed.batch,
        )

    def test_filter_3(self):
        dataset_1 = self.dataset.filter(
            lambda example, index: example["a"]["e"] > 2,
            with_indices=True,
        )

        dataset_2 = dataset_1.filter(
            lambda example, index: example["a"]["e"] > 3,
            with_indices=True,
        )

        # Dataset has the right columns
        self.assertEqual(set(dataset_1.column_names), {"a", "b", "i"})
        self.assertEqual(set(dataset_2.column_names), {"a", "b", "i"})

        # Datasets have the right data
        self.assertEqual(
            dataset_1[:],
            {"a": [{"e": 3}, {"e": 4}], "b": ["w", "x"], "i": self.testbed.images[2:]},
        )

        self.assertEqual(
            dataset_2[:],
            {"a": [{"e": 4}], "b": ["x"], "i": self.testbed.images[3:]},
        )

        # Original dataset is still the same
        self.assertEqual(
            self.dataset[:],
            self.testbed.batch,
        )

    def test_filter_add_column(self):
        dataset = self.dataset.filter(
            lambda example: example["a"] == {"e": 1},
            with_indices=False,
        )

        # Add b values back as another column 'c'
        dataset.add_column("c", dataset["b"])

        # Dataset has the right data
        self.assertEqual(
            dataset[:],
            {"a": [{"e": 1}], "b": ["u"], "c": ["u"], "i": [self.testbed.images[0]]},
        )

    def test_remove_column(self):
        self.dataset.remove_column("b")
        self.dataset.remove_column("i")
        self.assertEqual(self.dataset.column_names, ["a"])
