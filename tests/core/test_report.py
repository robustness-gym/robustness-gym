"""Unittests for Report."""
from unittest import TestCase, skip

import pandas as pd

from robustnessgym.report.report import (
    ClassDistributionColumn,
    NumericColumn,
    Report,
    ScoreColumn,
)


class TestReport(TestCase):
    def setUp(self):
        self.cols = [
            ScoreColumn("f1", 0, 1, is_0_to_1=True),
            ScoreColumn("perplexity", 0, 50),
            ClassDistributionColumn("Class Dist", ["e", "n", "c"]),
            NumericColumn("Size"),
        ]
        self.data = pd.DataFrame(
            [
                ["Cat A", "Slice C", 0.1, 5, [0.1, 0.2, 0.7], 300],
                ["Cat C", "Slice A", 0.2, 10, [0.4, 0.2, 0.4], 3],
                ["Cat A", "Slice A", 0.3, 15, [0.1, 0, 0.9], 5000],
                ["Cat B", "Slice B", 0.4, 20, [0.5, 0.4, 0.1], 812],
                ["Cat B", "Slice D", 0.5, 25, [0.3, 0.2, 0.5], 13312],
            ]
        )
        self.model_name = "BERT"
        self.dataset_name = "SNLI"

    def test_init(self):
        # Create a basic report
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        self.assertTrue(self.data.equals(report.data))

        # Pass config params
        custom_color_scheme = ["#000000"]
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            color_scheme=custom_color_scheme,
        )
        self.assertEqual(custom_color_scheme, report.config["color_scheme"])

    def test_sort(self):
        # Sort alphabetically
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        report.sort()
        actual = report.data
        expected = pd.DataFrame(
            [
                ["Cat A", "Slice A", 0.3, 15, [0.1, 0, 0.9], 5000],
                ["Cat A", "Slice C", 0.1, 5, [0.1, 0.2, 0.7], 300],
                ["Cat B", "Slice B", 0.4, 20, [0.5, 0.4, 0.1], 812],
                ["Cat B", "Slice D", 0.5, 25, [0.3, 0.2, 0.5], 13312],
                ["Cat C", "Slice A", 0.2, 10, [0.4, 0.2, 0.4], 3],
            ]
        )
        self.assertTrue(actual.equals(expected))

        # Sort by specified category order
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        report.sort(
            category_order={
                "Cat B": 0,
                "Cat C": 2,
                "Cat A": 1,
            }
        )
        actual = report.data
        expected = pd.DataFrame(
            [
                ["Cat B", "Slice B", 0.4, 20, [0.5, 0.4, 0.1], 812],
                ["Cat B", "Slice D", 0.5, 25, [0.3, 0.2, 0.5], 13312],
                ["Cat A", "Slice A", 0.3, 15, [0.1, 0, 0.9], 5000],
                ["Cat A", "Slice C", 0.1, 5, [0.1, 0.2, 0.7], 300],
                ["Cat C", "Slice A", 0.2, 10, [0.4, 0.2, 0.4], 3],
            ]
        )
        self.assertTrue(actual.equals(expected))

        # Sort by specified slice order
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        report.sort(
            slice_order={"Slice D": 0, "Slice C": 1, "Slice B": 2, "Slice A": 3}
        )
        actual = report.data
        expected = pd.DataFrame(
            [
                ["Cat A", "Slice C", 0.1, 5, [0.1, 0.2, 0.7], 300],
                ["Cat A", "Slice A", 0.3, 15, [0.1, 0, 0.9], 5000],
                ["Cat B", "Slice D", 0.5, 25, [0.3, 0.2, 0.5], 13312],
                ["Cat B", "Slice B", 0.4, 20, [0.5, 0.4, 0.1], 812],
                ["Cat C", "Slice A", 0.2, 10, [0.4, 0.2, 0.4], 3],
            ]
        )
        self.assertTrue(actual.equals(expected))

        # Sort by specified category order and slice order
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        report.sort(
            category_order={
                "Cat B": 0,
                "Cat C": 2,
                "Cat A": 1,
            },
            slice_order={"Slice D": 0, "Slice C": 1, "Slice B": 2, "Slice A": 3},
        )
        actual = report.data
        expected = pd.DataFrame(
            [
                ["Cat B", "Slice D", 0.5, 25, [0.3, 0.2, 0.5], 13312],
                ["Cat B", "Slice B", 0.4, 20, [0.5, 0.4, 0.1], 812],
                ["Cat A", "Slice C", 0.1, 5, [0.1, 0.2, 0.7], 300],
                ["Cat A", "Slice A", 0.3, 15, [0.1, 0, 0.9], 5000],
                ["Cat C", "Slice A", 0.2, 10, [0.4, 0.2, 0.4], 3],
            ]
        )
        self.assertTrue(actual.equals(expected))

    def test_filter(self):
        # Filter by category
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        report.filter(categories=["Cat B"])
        actual = report.data
        expected = pd.DataFrame(
            [
                ["Cat B", "Slice B", 0.4, 20, [0.5, 0.4, 0.1], 812],
                ["Cat B", "Slice D", 0.5, 25, [0.3, 0.2, 0.5], 13312],
            ]
        )
        self.assertTrue(actual.equals(expected))

        # Filter by slice
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        report.filter(slices=["Slice A", "Slice C"])
        actual = report.data
        expected = pd.DataFrame(
            [
                ["Cat A", "Slice C", 0.1, 5, [0.1, 0.2, 0.7], 300],
                ["Cat C", "Slice A", 0.2, 10, [0.4, 0.2, 0.4], 3],
                ["Cat A", "Slice A", 0.3, 15, [0.1, 0, 0.9], 5000],
            ]
        )
        self.assertTrue(actual.equals(expected))

    def test_rename(self):
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        category_map = {"Cat C": "Cat D"}
        slice_map = {"Slice A": "Slice D"}
        report.rename(category_map=category_map, slice_map=slice_map)
        actual = report.data
        expected = pd.DataFrame(
            [
                ["Cat A", "Slice C", 0.1, 5, [0.1, 0.2, 0.7], 300],
                ["Cat D", "Slice D", 0.2, 10, [0.4, 0.2, 0.4], 3],
                ["Cat A", "Slice D", 0.3, 15, [0.1, 0, 0.9], 5000],
                ["Cat B", "Slice B", 0.4, 20, [0.5, 0.4, 0.1], 812],
                ["Cat B", "Slice D", 0.5, 25, [0.3, 0.2, 0.5], 13312],
            ]
        )
        self.assertTrue(actual.equals(expected))

    def test_set_class_codes(self):
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        custom_class_codes = ["A", "B", "C"]
        report.set_class_codes(custom_class_codes)
        for col in report.columns:
            if isinstance(col, ClassDistributionColumn):
                self.assertEqual(col.class_codes, custom_class_codes)

    def test_set_range(self):
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        report.set_range("f1", 0.1, 0.3)
        for col in report.columns:
            if col.title == "f1":
                self.assertEqual((col.min_val, col.max_val), (0.1, 0.3))

    def test_figure(self):
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )

        # Original unsorted data should cause an error
        self.assertRaises(ValueError, report.figure)

        # Sort should resolve that error
        report.sort()
        try:
            report.figure()
        except ValueError:
            self.fail("report.figure() raised ValueError unexpectedly!")

    @skip("Manual test")
    def test_display(self):
        report = Report(
            self.data,
            self.cols,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )

        report.sort()
        figure = report.figure()
        figure.show()

        report.sort(category_order={"Cat C": 1, "Cat A": 2, "Cat B": 3})
        report.rename(slice_map={"Slice A": "A"}, category_map={"Cat B": "B"})
        report.filter(slices=["A", "Slice B", "Slice C"])
        report.set_range("f1", 0.05, 0.45)
        report.update_config(font_size_heading=16)
        figure = report.figure(show_title=True)
        figure.show()

    @skip("Manual test")
    def test_display_2(self):
        data = pd.DataFrame(
            [
                [
                    "Eval",
                    "snli1",
                    0.8799999952316284,
                    0.876409113407135,
                    [0.368, 0.304, 0.328],
                    [0.344, 0.288, 0.368],
                    125,
                ],
                [
                    "Eval",
                    "snli2",
                    0.8799999952316284,
                    0.876409113407135,
                    [0.368, 0.304, 0.328],
                    [0.344, 0.288, 0.368],
                    125,
                ],
                [
                    "Eval",
                    "snli3",
                    0.8799999952316284,
                    0.876409113407135,
                    [0.368, 0.304, 0.328],
                    [0.344, 0.288, 0.368],
                    125,
                ],
            ]
        )
        cols = [
            ScoreColumn("F1", min_val=0, max_val=1, is_0_to_1=True),
            ScoreColumn("Accuracy", min_val=0, max_val=1, is_0_to_1=True),
            ClassDistributionColumn("Class Dist", ["e", "n", "c"]),
            ClassDistributionColumn("Pred Dist", ["e", "n", "c"]),
            NumericColumn("Size"),
        ]
        report = Report(data, cols)
        report.figure().show()
