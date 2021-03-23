"""Unittests for metrics."""
from unittest import TestCase

import numpy as np
import torch
from sklearn.metrics import f1_score

from robustnessgym.core.metrics import accuracy, f1
from tests.testbeds import MockTestBedv0


class TestSlice(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_accuracy_1(self):
        # Create some data
        predictions = [0, 1, 1, 0, 1, 2, 3, 7]
        labels = [1, 0, 0, 0, 1, 2, 4, 8]

        # Ground-truth score
        gt_score = np.mean([(p == l) for p, l in zip(predictions, labels)])

        # Accuracy using lists
        score = accuracy(predictions, labels)
        self.assertEqual(score, gt_score)

        # Accuracy using np.ndarray
        score = accuracy(np.array(predictions), np.array(labels))
        self.assertEqual(score, gt_score)

        # Accuracy using torch.tensor
        score = accuracy(torch.tensor(predictions), torch.tensor(labels))
        self.assertEqual(score, gt_score)

    def test_accuracy_2(self):
        # Create some data
        predictions = []
        labels = []

        # Accuracy using lists
        score = accuracy(predictions, labels)
        self.assertTrue(np.isnan(score))

        # Accuracy using np.ndarray
        score = accuracy(np.array(predictions), np.array(labels))
        self.assertTrue(np.isnan(score))

        # Accuracy using torch.tensor
        score = accuracy(torch.tensor(predictions), torch.tensor(labels))
        self.assertTrue(np.isnan(score))

    def test_accuracy_3(self):
        # Create some data
        predictions = [1, 2]
        labels = [1]

        # Mismatched lengths
        with self.assertRaises(ValueError):
            accuracy(predictions, labels)

    def test_f1_1(self):
        # Create some data
        predictions = [0, 1, 1, 0, 1, 2, 3, 7]
        labels = [1, 0, 0, 0, 1, 2, 4, 8]

        with self.assertRaises(ValueError):
            # F1 using lists
            f1(predictions, labels)

        with self.assertRaises(ValueError):
            # F1 using np.ndarray
            f1(np.array(predictions), np.array(labels))

        with self.assertRaises(ValueError):
            # F1 using torch.tensor
            f1(torch.tensor(predictions), torch.tensor(labels))

    def test_f1_2(self):
        # Create some data
        predictions = []
        labels = []

        # Ground-truth score
        gt_score = f1_score(y_true=labels, y_pred=predictions)

        # F1 using lists
        score = f1(predictions, labels)
        self.assertEqual(score, gt_score)

        # F1 using np.ndarray
        score = f1(np.array(predictions), np.array(labels))
        self.assertEqual(score, gt_score)

        # F1 using torch.tensor
        score = f1(torch.tensor(predictions), torch.tensor(labels))
        self.assertEqual(score, gt_score)

    def test_f1_3(self):
        # Create some data
        predictions = [1, 2]
        labels = [1]

        # Mismatched lengths
        with self.assertRaises(ValueError):
            f1(predictions, labels)
