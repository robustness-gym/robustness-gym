from __future__ import annotations

import pathlib
from typing import *

import dill
import pandas as pd
from fuzzywuzzy import process
from tqdm import tqdm

from robustnessgym.core.model import Model
from robustnessgym.core.report import Report, ScoreColumn, NumericColumn, ClassDistributionColumn
from robustnessgym.core.slice import Slice
from robustnessgym.tasks.task import Task
from robustnessgym.core.tools import persistent_hash

from robustnessgym.core.constants import GENERIC, SUBPOPULATION, ATTACK, AUGMENTATION, CURATION

TEST = 'test'
category_to_label= {
    TEST: 'Test Set',
    GENERIC: 'Slice',
    SUBPOPULATION: 'SubPop',
    ATTACK: 'Attack',
    AUGMENTATION: 'Augment',
    CURATION: 'Eval'
}
category_to_order = {
    TEST: 0,
    SUBPOPULATION: 1,
    AUGMENTATION: 2,
    CURATION: 3,
    ATTACK: 4,
    GENERIC: 5
}


# TODO(karan): make the TestBench hashable
class TestBench:

    def __init__(self,
                 identifier: str,
                 task: Task,
                 slices: Collection[Slice],
                 dataset_id: str = None):

        # An identifier for the TestBench
        self.identifier = identifier

        # Set the task
        self.task = task

        # Create the collection of slices
        self.slices = set()
        self.slice_identifiers = set()
        self._slice_table = {}
        self.add_slices(slices)

        # The testbench internally tracks metrics
        self.metrics = {}

        # The schema tells the testbench which columns to extract from the slices for evaluation
        self.schema_type = 'default'

        self.dataset_id = dataset_id

    @property
    def version(self):
        """
        The test bench version.

        Returns: version

        """
        return '1.0.0'

    @classmethod
    def for_dataset(
            cls,
            dataset: str,
            task: Optional[Union[str, Task]] = None,
            version: str = None):
        """
        Create a test bench for a dataset.

        Args:
            dataset:
            task:
            version:

        Returns:

        """

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

    def _human_readable_identifiers(self):
        # Temporary function to generate human readable names
        groups = {}
        for ident in self.slice_identifiers:
            if "->" in str(ident):
                builder_ident = str(ident).split(" -> ")[-1]
                builder_ident, cols = builder_ident.split(" @ ")
                name = builder_ident.split("(")[0]
                if name not in groups:
                    groups[name] = set()
                groups[name].add((builder_ident, cols))

        group_info = {}
        for key, group in groups.items():
            if len(group) == 1:
                group_info[key] = 'name'
            else:
                only_single_column = len(set([t[1] for t in group])) == 1
                if only_single_column:
                    group_info[key] = 'builder_ident'
                else:
                    group_info[key] = 'full'

        ident_mapping = {}
        for ident in self.slice_identifiers:
            if "->" in str(ident):
                builder_ident = str(ident).split(" -> ")[-1]
                builder_ident, cols = builder_ident.split(" @ ")
                name = builder_ident.split("(")[0]

                if group_info[name] == 'name':
                    new_ident = name
                elif group_info[name] == 'builder_ident':
                    new_ident = builder_ident
                elif group_info[name] == 'full':
                    new_ident = str(ident).split(" -> ")[-1]

                if new_ident.startswith('NlpAugTransformation'):
                    new_ident = new_ident.split("NlpAugTransformation(pipeline=[")[1].split("])")[0]

            else:
                new_ident = str(ident).split("(")[0]

            ident_mapping[ident] = new_ident

        self.ident_mapping = ident_mapping

    def add_slices(self,
                   slices: Collection[Slice]):
        """
        Add slices to the testbench.

        Args:
            slices: collection of Slice objects

        Returns:

        """
        if isinstance(slices, Slice):
            slices = [slices]

        # Only add slices that aren't already present in the testbench and have non-zero length
        for sl in slices:
            if sl.identifier not in self.slice_identifiers and len(sl) > 0:
                self.slices.add(sl)
                self.slice_identifiers.add(sl.identifier)
                self._slice_table[sl.identifier] = sl

    def evaluate(self,
                 model: Model,
                 batch_size: int = 32,
                 coerce_fn: Callable = None) -> Dict:
        """
        Evaluate a model using the test bench.

        Args:
            model: model to evaluate
            batch_size: batch size for inference
            coerce_fn: function to coerce the model's outputs. Useful if the model's outputs cannot directly be compared
            to the targets.

        Returns: dict mapping slice identifiers to evaluation metrics.

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

        # Store the model_metrics
        if model.identifier not in self.metrics:
            self.metrics[model.identifier] = {}

        # Run the model on all the slices
        # TODO(karan): For slices that are subpopulations, the same example can be in multiple slices
        #  and will be run through the model multiple times. Create a UnionSlice?
        for sl in tqdm(self.slices):
            if sl.identifier not in self.metrics[model.identifier]:
                # Evaluate on the slice
                self.metrics[model.identifier][sl.identifier] = model.evaluate(
                    dataset=sl,
                    input_columns=self.task.input_schema.keys(),
                    output_columns=self.task.output_schema.keys(),
                    batch_size=batch_size,
                    coerce_fn=coerce_fn
                )

        return self.metrics[model.identifier]

    def create_report(self,
                      model: Model,
                      batch_size: int = 32,
                      coerce_fn: Callable = None,
                      metric_ids: List[str] = None) -> Report:
        """
        Generate a report for a model.
        """

        # Grab the metrics
        # TODO(karan): ask the model to return side-information (probs, logits, embeddings)
        model_metrics = self.evaluate(model=model, batch_size=batch_size, coerce_fn=coerce_fn)

        # Create a consolidated "report"

        # TODO(karan): make this more general
        self._human_readable_identifiers()

        data = []
        for slice in self.slices:
            row = {
                'category_order': category_to_order[slice.category],
                'category': category_to_label.get(slice.category, slice.category.capitalize()),
                'slice_name': self.ident_mapping[slice.identifier],  # str(slice.identifier),
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

    def set_schema(self,
                   schema_type: str):
        assert schema_type in {'default', 'task'}

        if self.schema_type == schema_type:
            return

        if schema_type == 'task':
            self.slices = {self.task.remap_schema(slice) for slice in self.slices}
            self.schema_type = schema_type
        elif schema_type == 'default':
            # TODO(karan): undo the schema standardization
            raise NotImplementedError

    def search(self,
               keyword: str,
               limit: int = 3):
        return [self._slice_table[t[0]] for t in process.extract(keyword,
                                                                 self.slice_identifiers,
                                                                 limit=limit)]

    def save(self,
             path: str) -> None:
        """
        Save the current testbench to disk. This will save all slices in the testbench to disk, as well as metrics
        and other metadata associated with this testbench.

        Args:
            path: string path to the save directory

        Returns: None

        >>> testbench = TestBench(identifier='my-testbench', task=TernaryNaturalLanguageInference())
        # Save to the current directory
        >>> testbench.save('.')
        # Load back the testbench
        >>> testbench = TestBench.load('my-testbench')

        """

        # Path to the save directory
        savedir = pathlib.Path(path) / f'{self.identifier}'

        # Create a directory inside savedir for the slices
        (savedir / 'slices').mkdir(parents=True, exist_ok=True)

        # Save all the slices
        pbar = tqdm(self.slices)
        for sl in pbar:
            pbar.set_description(f"Saving slice {str(sl.identifier)[:100]}...")
            sl.save_to_disk(str(savedir / 'slices' / str(persistent_hash(str(sl.identifier)))))

        # Save metrics
        dill.dump(self.metrics, open(str(savedir / 'metrics.dill'), 'wb'))

        # Save metadata
        dill.dump({
            'task': self.task,
            'identifier': self.identifier,
            'dataset_id': self.dataset_id,
        },
            open(str(savedir / 'metadata.dill'), 'wb')
        )

    @classmethod
    def available(cls,
                  path: str) -> List[str]:
        """
        Check the list of available testbenches in a directory.

        Args:
            path: string path to a directory. The testbenches available inside this directory will be returned.

        Returns: list of available testbenches

        """

        # Path to the save directory
        savedir = pathlib.Path(path)

        # Loop over the folders
        testbench_identifiers = []
        for maybe_testbench in savedir.glob('*'):
            if maybe_testbench.is_dir() and (maybe_testbench / 'metadata.dill').exists():
                testbench_identifiers.append(maybe_testbench.name)

        return testbench_identifiers

    @classmethod
    def load(cls,
             path: str) -> TestBench:
        """
        Load a testbench from disk.

        Args:
            path: string path to the testbench directory

        Returns:

        """

        # Path to the save directory
        savedir = pathlib.Path(path)

        # Load all the slices
        slices = []
        for sl_path in tqdm(list((savedir / 'slices').glob('*'))):
            try:
                slices.append(Slice.load_from_disk(str(sl_path)))
            except FileNotFoundError:
                continue

        # Load metrics
        metrics = dill.load(open(str(savedir / 'metrics.dill'), 'rb'))

        # Load metadata
        metadata = dill.load(
            open(str(savedir / 'metadata.dill'), 'rb')
        )

        # Create the testbench
        testbench = cls(
            identifier=metadata['identifier'],
            task=metadata['task'],
            slices=slices,
        )

        # Set previously stored metrics
        testbench.metrics = metrics

        return testbench

    def make(self,
             identifier: str):
        # Resolve the location of the TestBench

        # Pull the TestBench
        return self.pull(identifier)

    def pull(self,
             identifier: str):
        pass

    def publish(self):
        pass
