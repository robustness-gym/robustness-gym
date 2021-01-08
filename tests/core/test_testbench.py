"""Unittests for TestBench
   isort:skip_file
"""

import functools
from unittest import TestCase, skip

from robustnessgym import Dataset, TestBench, Slice, Task
from robustnessgym.core.model import Model

# fmt: off


class TestTestbench(TestCase):
    # def setUp(self):
    #     self.testbed = MockTestBedv0()

    @skip("Long-running test")
    def test_create_report(self):

        # Create task
        task_identifier = "TernaryNaturalLanguageInference"
        task = Task.create(task=task_identifier)

        # Create model
        model_identifier = "textattack/bert-base-uncased-snli"
        model = Model.huggingface(
            identifier=model_identifier,
            task=task,
        )

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

        # Display the report
        report = testbench.create_report(
            model=model,
            coerce_fn=functools.partial(Model.remap_labels, label_map=[1, 2, 0]),
        )
        report.figure().show()
