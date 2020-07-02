"""Unittests for TestBench."""

import functools
from unittest import TestCase, skip

import torch

from robustnessgym import Dataset, Slice, Task, TestBench
from robustnessgym.core.model import Model


class TestTestbench(TestCase):
    # TODO add NLG test cases

    @skip("Long-running test")
    def test_evaluate(self):

        # Test evaluate with a task
        task = self._get_task()
        testbench = self._get_testbench(task)
        model = self._get_model(task=task)
        testbench.evaluate(
            model=model,
            coerce_fn=functools.partial(Model.remap_labels, label_map=[1, 2, 0]),
        )
        self.assertIn(model.identifier, testbench.metrics)
        self.assertSetEqual(
            set(testbench.metrics[model.identifier].keys()),
            set(sl.identifier for sl in testbench.slices),
        )
        for sl in testbench.slices:
            eval_dict = testbench.metrics[model.identifier][sl.identifier]
            self.assertSetEqual(set(eval_dict.keys()), set(testbench.task.metrics))
            for value in eval_dict.values():
                self.assertIsNotNone(value)

        # Test evaluate without a task

        testbench = self._get_testbench()
        model = self._get_model(is_classifier=True)
        # Check that it raises exception if input_columns,
        # output_columns not specified in absence of task
        self.assertRaises(
            ValueError,
            testbench.evaluate,
            model=model,
            coerce_fn=functools.partial(Model.remap_labels, label_map=[1, 2, 0]),
        )

        task = self._get_task()
        testbench = self._get_testbench(task=task)
        model = self._get_model(task=task)
        testbench.evaluate(
            model=model,
            coerce_fn=functools.partial(Model.remap_labels, label_map=[1, 2, 0]),
            input_columns=["sentence1", "sentence2"],
            output_columns=["label"],
        )
        self.assertIn(model.identifier, testbench.metrics)
        self.assertSetEqual(
            set(testbench.metrics[model.identifier].keys()),
            set(sl.identifier for sl in testbench.slices),
        )
        for sl in testbench.slices:
            eval_dict = testbench.metrics[model.identifier][sl.identifier]
            self.assertSetEqual(set(eval_dict.keys()), set(testbench.task.metrics))
            for value in eval_dict.values():
                self.assertIsNotNone(value)

    @skip("Long-running test")
    def test_add_metrics(self):
        testbench = self._get_testbench()
        metrics = {"snli_1": {"f1": 0.1, "accuracy": 0.3}}
        testbench.add_metrics("bert", metrics)
        self.assertEqual(testbench.metrics["bert"], metrics)

    @skip("Long-running test")
    def test_add_predictions(self):
        model = "bert-base"
        task = self._get_task()
        testbench = self._get_testbench(task)
        torch.manual_seed(1)
        predictions = {}
        for sl in testbench.slices:
            predictions[sl.identifier] = torch.randint(high=3, size=(len(sl),))

        testbench.add_predictions(model=model, predictions=predictions)

        self.assertIn(model, testbench.metrics)
        self.assertSetEqual(
            set(testbench.metrics[model].keys()),
            set(sl.identifier for sl in testbench.slices),
        )
        for sl in testbench.slices:
            eval_dict = testbench.metrics[model][sl.identifier]
            self.assertSetEqual(set(eval_dict.keys()), set(testbench.task.metrics))
            for value in eval_dict.values():
                self.assertIsNotNone(value)

    @skip("Long-running test")
    def test_create_report(self):
        task = self._get_task()
        testbench = self._get_testbench(task=task)
        model = self._get_model(task=task)
        testbench.evaluate(
            model=model,
            coerce_fn=functools.partial(Model.remap_labels, label_map=[1, 2, 0]),
        )
        report = testbench.create_report(model)
        fig = report.figure()
        self.assertIsNotNone(fig)

    @skip("Manual test")
    def test_display_report(self):

        # # Create report using 'evaluate'
        task = self._get_task()
        testbench = self._get_testbench(task=task)
        model = self._get_model(task=task)
        testbench.evaluate(
            model=model,
            coerce_fn=functools.partial(Model.remap_labels, label_map=[1, 2, 0]),
        )
        report = testbench.create_report(model)
        fig = report.figure()
        fig.show()

        # Create report using add_predictions
        task = self._get_task()
        testbench = self._get_testbench(task)
        torch.manual_seed(1)
        predictions = {}
        for sl in testbench.slices:
            predictions[sl.identifier] = torch.randint(high=3, size=(len(sl),))
        testbench.add_predictions(model="bert-base", predictions=predictions)
        report = testbench.create_report("bert-base")
        fig = report.figure()
        fig.show()

        # Create report using add_metrics
        testbench = self._get_testbench()
        metrics = {
            "snli_1": {"f1": 0.1, "accuracy": 0.1},
            "snli_2": {"f1": 0.5, "accuracy": 0.5},
            "snli_3": {"f1": 0.9, "accuracy": 0.4},
        }
        testbench.add_metrics(model, metrics)
        report = testbench.create_report(model, metric_ids=["f1", "accuracy"])
        fig = report.figure()
        fig.show()

    def _get_task(self):
        # Create task
        task_identifier = "TernaryNaturalLanguageInference"
        task = Task.create(task=task_identifier)
        return task

    def _get_model(self, **kwargs):
        # TODO have a proper mock model
        # Create model
        model_identifier = "textattack/bert-base-uncased-snli"
        model = Model.huggingface(identifier=model_identifier, **kwargs)
        return model

    def _get_testbench(self, task=None):
        # TODO Have a proper mock testbench
        # Create test bench
        testbench_identifier = "test-testbench"
        testbench = TestBench(
            identifier=testbench_identifier,
            task=task,
            slices=[
                Slice(
                    dataset=Dataset.load_dataset("snli", split="train[:128]"),
                    identifier="snli_1",
                ).filter(lambda example: example["label"] != -1),
                Slice(
                    dataset=Dataset.load_dataset("snli", split="validation[:128]"),
                    identifier="snli_2",
                ).filter(lambda example: example["label"] != -1),
                Slice(
                    dataset=Dataset.load_dataset("snli", split="test[:128]"),
                    identifier="snli_3",
                ).filter(lambda example: example["label"] != -1),
            ],
            dataset_id="snli",
        )
        return testbench
