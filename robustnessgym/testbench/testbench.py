from typing import *

import pandas as pd

from robustnessgym.model import Model
from robustnessgym.report import Report, ScoreColumn, NumericColumn, ClassDistributionColumn
from robustnessgym.slice import Slice
from robustnessgym.tasks.task import Task


# TODO(karan): make the TestBench hashable
class TestBench:

    def __init__(self,
                 identifier: str,
                 task: Task,
                 slices: List[Slice],
                 dataset_id: str = None):

        # An identifier for the TestBench
        self.identifier = identifier

        self.task = task

        self.slices = slices

        self.metrics = {}

        self.schema_type = 'default'

        self.dataset_id = dataset_id

    @classmethod
    def for_dataset(
            cls,
            dataset: str,
            task: Optional[Union[str, Task]] = None,
            version: str = None,
    ):
        # Infer the task from the dataset
        inferred_task = Task.lookup(dataset=dataset)()

        # Check that the inferred task matches the task argument
        if task is not None and task != inferred_task:
            raise AssertionError(f"Dataset {dataset} is only compatible with {inferred_task}, not {task}.")

        return TestBench(
            identifier=f'{dataset}-{task}-{version}',
            task=inferred_task,
            slices=[],
        )

    @classmethod
    def for_task(
            cls,
            task: Union[str, Task],
            version: str = None,
    ):
        return TestBench(
            identifier=f'{task}-{version}',
            task=task,
            slices=[],
        )

    @property
    def version(self):
        return '1.0.0'

    def evaluate(self,
                 model: Model,
                 batch_size: int = 32,
                 coerce_fn: Callable = None) -> Dict:
        """

        """

        # Set the schema using the task
        self.set_schema('task')

        # TODO(karan): Uncomment and fix this assert on the type of outputs that model(..) returns
        # # Grab 2 examples from the first slice, run it through the model and check that the output is a dictionary
        # output = model(dataset=Dataset.from_batch(self.slices[0][:2]),
        #                input_keys=self.task.input_schema.keys(),
        #                output_keys=self.task.output_schema.keys(),
        #                batch_size=2,
        #                coerce_fn=coerce_fn)
        # print(output)
        # assert isinstance(output, Sequence) and isinstance(output[0], Mapping), \
        #     "model(..) must return a list of dictionaries. Each dictionary should map metric names to values."

        # Run the model on all the slices
        # TODO(karan): For slices that are subpopulations, the same example can be in multiple slices
        #  and will be run through the model multiple times. Create a UnionSlice?

        model_metrics = {slice.identifier: None for slice in self.slices}

        for slice in self.slices:
            # Evaluate on the slice
            model_metrics[slice.identifier] = model.evaluate(
                dataset=slice,
                input_keys=self.task.input_schema.keys(),
                output_keys=self.task.output_schema.keys(),
                batch_size=batch_size,
                coerce_fn=coerce_fn
            )

        # Store the model_metrics
        self.metrics[model.identifier] = model_metrics

        return model_metrics

    def create_report(self,
                      model: Model,
                      batch_size: int = 32,
                      coerce_fn: Callable = None,
                      metric_ids: List[str] = None) -> Report:
        """
        Generate a report for a model.
        """
        if model.identifier not in self.metrics:
            # TODO(karan): ask the model to return side-information (probs, logits, embeddings)
            self.evaluate(model=model, batch_size=batch_size, coerce_fn=coerce_fn)

        # Grab the metrics
        model_metrics = self.metrics[model.identifier]

        # Create a consolidated "report"
        # TODO(karan): these should be constants somewhere in Curator
        categories = [
            'subpopulation',
            'augmentation',
            'attack',
            'curated',
        ]
        category_to_index = {category: i for i, category in enumerate(categories)}

        df = pd.DataFrame()
        data = []
        for slice in self.slices:
            row = {
                'category_order': category_to_index[slice.category],
                'category': slice.category,
                'slice_name': slice.identifier,
                'Size': len(slice)
            }
            slice_metrics = model_metrics[slice.identifier]
            row.update(slice_metrics)
            data.append(row)
            if metric_ids is None:
                metric_ids = list(slice_metrics.keys())
        df = pd.DataFrame(data)
        df = df.sort_values(['category_order', 'slice_name'])

        columns = []
        for metric_id in metric_ids:
            # TODO store these min and max values somewhere
            if metric_id in ('class_dist', 'pred_dist'):
                class_names = self.task.output_schema.features[list(self.task.output_schema.keys())[0]].names
                class_inits = [name[0].upper() for name in class_names]
                if len(set(class_inits)) == len(class_inits):
                    columns.append(ClassDistributionColumn(metric_id, class_inits))
            else:
                columns.append(ScoreColumn(metric_id, 0.0, 100.0))
        # if self.task.classification() and
        # columns.append(ClassDistributionColumn())
        columns.append(NumericColumn('Size'))
        # Aggregate: convert list of dicts -> dict with aggregated values
        # TODO(karan): generalize aggregation
        # slice_metrics = tz.merge_with(np.mean, slice_metrics)
        # Task-dependent model predictions
        # TODO(karan): e.g. average class distribution predicted, figure out how to put this in
        # Task-dependent slice information
        # TODO(karan): e.g. class distribution

        return Report(df, columns, model.identifier, self.dataset_id)

    def set_schema(self, schema_type: str):
        assert schema_type in {'default', 'task'}

        if self.schema_type == schema_type:
            return

        if schema_type == 'task':
            self.slices = [self.task.remap_schema(slice) for slice in self.slices]
            self.schema_type = schema_type
        elif schema_type == 'default':
            # TODO(karan): undo the schema standardization
            raise NotImplementedError

    def save(self, path):
        pass

    def load(self, path):
        pass

    def make(self,
             identifier: str):
        # Resolve the location of the TestBench

        # Pull the TestBench
        return self.pull(identifier)

    def pull(self, identifier: str):
        pass

    def publish(self):
        pass
