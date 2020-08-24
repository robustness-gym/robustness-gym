import sys
import cytoolz as tz
import numpy as np
import functools
from typing import *
import torch

sys.path.append('..')

from robustness_gym import *

from transformers import *

# Integration for the Robustness Report
# Assumes
# - user elected to do (model, dataset) report generation (and not (model, task))
# - user selected model = textattack/bert-base-uncased-snli
# - user selected dataset = snli

# Load the task
nli = TernaryNaturalLanguageInference()

# Load the SNLI dataset
snli_dataset = Dataset.load_dataset('snli', split='train[:1%]')

# Load the SNLI model
snli_model = Model.huggingface(
    identifier="textattack/bert-base-uncased-snli",
    task=nli,
)

# Evaluate the model on the dataset slice
snli_model.evaluate(
    dataset=snli_dataset,
    # Tokenize and pass the premise and hypothesis as input to the model
    input_keys=['premise', 'hypothesis'],
    # Use the label to evaluate the models predictions
    output_keys=['label'],
    batch_size=32,
    # The outputs of the model need to be permuted to match the SNLI labels
    coerce_fn=functools.partial(Model.remap_labels, label_map=[1, 2, 0])
)


def generate_report(model: Model,
                    dataset: Optional[str] = None,
                    task: Optional[str] = None,
                    version: Optional[str] = None,
                    testbench: TestBench = None):
    """
    Create a robustness report for a model.
    """

    if testbench is None:

        if dataset is not None:
            # Create the default TestBench for the dataset
            testbench = TestBench.for_dataset(
                dataset=dataset,
                task=task,
                version=version
            )
        elif task is not None:
            # Create the default TestBench for the task
            testbench = TestBench.for_task(
                task=task,
                version=version
            )

    # Generate a report
    return testbench.create_report(model=model)


# Create a TestBench
testbench = TestBench(
    identifier='snli-nli-0.0.1dev',
    task=nli,
    slices=[
        Slice.from_dataset(identifier='snli-train',
                           dataset=snli_dataset),
    ]
)

# Generate report for SNLI
report = generate_report(
    model=snli_model,
    dataset='snli',
)

# Send report to be visualized
# ...
