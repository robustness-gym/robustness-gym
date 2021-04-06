from __future__ import annotations

import inspect
import json
import logging
import pathlib
from typing import Callable, Collection, Dict, List, Optional, Sequence, Union

import dill
import pandas as pd
import torch
from fuzzywuzzy import process
from tqdm import tqdm

from robustnessgym.core.constants import (
    ATTACK,
    AUGMENTATION,
    CURATION,
    GENERIC,
    SUBPOPULATION,
)
from robustnessgym.core.dataset import Dataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.metrics import compute_metric, get_metric
from robustnessgym.core.model import Model
from robustnessgym.core.report import (
    ClassDistributionColumn,
    NumericColumn,
    Report,
    ScoreColumn,
)
from robustnessgym.core.slice import Slice
from robustnessgym.core.tools import persistent_hash
from robustnessgym.core.version import SemanticVersionerMixin
from robustnessgym.tasks.schema import Schema
from robustnessgym.tasks.task import Task

logger = logging.getLogger(__name__)


class DevBench(SemanticVersionerMixin):
    def __init__(
        self,
        dataset: Dataset,
    ):
        # Call the superclass
        super(DevBench, self).__init__()

        # An identifier for the DevBench
        self.identifier = Identifier("DevBench", dataset=str(dataset.identifier))

        # Dataset that the devbench operates on
        self._dataset = dataset

        # Create the collection of slices
        self._slices = set()
        self._slice_identifiers = set()
        self._slice_table = {}

        # The devbench has aggregators
        self.aggregators = {}

        # The devbench internally tracks metrics
        self.metrics = {}

        # Add slices if any
        self.add_slices(dataset)

    @property
    def dataset(self):
        """Dataset for the devbench."""
        return self._dataset

    @property
    def slices(self):
        """Slices in the devbench."""
        return self._slices

    @property
    def models(self):
        """Models in the devbench."""
        return list(self.aggregators.keys())

    def __repr__(self):
        return f"{self.identifier}(slices={len(self.slices)})"

    def _digest(self) -> str:
        return json.dumps([str(sl) for sl in self.slices])

    def add_slices(
        self,
        slices: Union[Dataset, Slice, Collection[Union[Dataset, Slice]]],
    ) -> None:
        """Add slices to the development bench.

        Args:
            slices: collection of Slice objects
        """
        if isinstance(slices, Slice) or isinstance(slices, Dataset):
            slices = [slices]

        # Add slices
        for sl in slices:
            if not isinstance(sl, Slice) and isinstance(sl, Dataset):
                # Convert `Dataset` to `Slice`
                sl = Slice(sl)

            if isinstance(sl, Slice):
                if (
                    sl.identifier not in self._slice_identifiers
                    and len(sl) > 0
                    and sl.lineage[0][1] == self.dataset.identifier
                ):
                    # Add slices that aren't present, have non-zero length and
                    # originate from the dataset
                    self._slices.add(sl)
                    self._slice_identifiers.add(sl.identifier)
                    self._slice_table[sl.identifier] = sl

        # Calculate all metrics
        self.calculate()

    def add_aggregators(
        self,
        aggregators: Dict[str, Dict[str, Callable]],
    ) -> None:
        """Add functions for aggregation, with a dictionary that maps a model
        name to aggregation functions."""

        # For each model
        for model, aggs in aggregators.items():
            # Iterate over the aggregation functions
            for agg_name, agg in aggs.items():
                assert isinstance(agg, Callable), "Aggregators must be functions."
                arguments = inspect.getfullargspec(agg).args
                assert len(arguments) == 1, "Aggregators must be single argument."
                assert arguments[0] == "dataset", (
                    "Aggregator argument name must be `dataset`, "
                    f"not `{arguments[0]}`."
                )

                # Store the aggregator
                if model not in self.aggregators:
                    self.aggregators[model] = {}
                self.aggregators[model][agg_name] = agg

        # Calculate all metrics
        self.calculate()

    def calculate(self):
        """Calculate all metrics that haven't been calculated yet."""
        # Iterate over all models
        for model in self.aggregators:

            # Add the model to the metrics dict
            if model not in self.metrics:
                self.metrics[model] = {}

            # Iterate over all slices
            for sl in self.slices:
                # Add the slice to the model's metrics dict
                if str(sl.identifier) not in self.metrics[model]:
                    self.metrics[model][str(sl.identifier)] = {}

                # Iterate over all aggregation functions
                for agg_name, aggregator in self.aggregators[model].items():
                    if agg_name not in self.metrics[model][str(sl.identifier)]:
                        # Aggregate
                        self.metrics[model][str(sl.identifier)][agg_name] = aggregator(
                            sl
                        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
    ) -> DevBench:
        """Create a DevBench from a dataset."""
        # Create the devbench
        devbench = DevBench(
            dataset=dataset,
        )

        return devbench

    def _human_readable_identifiers(self):
        # Temporary function to generate human readable names
        groups = {}
        for ident in self._slice_identifiers:
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
                group_info[key] = "name"
            else:
                only_single_column = len(set([t[1] for t in group])) == 1
                if only_single_column:
                    group_info[key] = "builder_ident"
                else:
                    group_info[key] = "full"

        ident_mapping = {}
        for ident in self._slice_identifiers:
            if "->" in str(ident):
                builder_ident = str(ident).split(" -> ")[-1]
                builder_ident, cols = builder_ident.split(" @ ")
                name = builder_ident.split("(")[0]

                if group_info[name] == "name":
                    new_ident = name
                elif group_info[name] == "builder_ident":
                    new_ident = builder_ident
                elif group_info[name] == "full":
                    new_ident = str(ident).split(" -> ")[-1]

                if new_ident.startswith("NlpAugTransformation"):
                    new_ident = new_ident.split("NlpAugTransformation(pipeline=[")[
                        1
                    ].split("])")[0]

            else:
                new_ident = str(ident).split("(")[0]

            ident_mapping[ident] = new_ident

        self.ident_mapping = ident_mapping

    def _common_aggregators(self, models: List[str]):

        common_aggregators = []
        # Iterate over all the models
        for model in models:
            assert model in self.metrics, f"Model {model} does not exist."
            common_aggregators.append(set([agg for agg in self.aggregators[model]]))

        # Find the common aggregators by taking an intersection
        return set.intersection(*[set(e) for e in common_aggregators])

    def create_report(
        self,
        models: List[str] = None,
    ) -> Report:
        """Generate report from cached metrics for a model
        Args:
            models: List of models.
        Returns:
            report
        """

        if len(self.slices) == 0:
            raise ValueError("Cannot create report for empty testbench")

        if models is not None:
            for model in models:
                assert model in self.metrics, f"Model {model} does not exist."
        else:
            # Use all the models that are available
            models = list(self.metrics.keys())

        # Set identifiers to be human readable
        self._human_readable_identifiers()

        # Get a list of aggregators that are common to `models`
        common_aggregators = list(self._common_aggregators(models))

        # Populate columns
        columns = []
        for model in models:
            for aggregator in common_aggregators:
                columns.append(
                    ScoreColumn(
                        f"{model}-{aggregator}",
                        min_val=0,
                        max_val=1,
                        is_0_to_1=True,
                    )
                )
        columns.append(NumericColumn("Size"))

        category_names = {
            GENERIC: "Slice",
            SUBPOPULATION: "SubPop",
            ATTACK: "Attack",
            AUGMENTATION: "Augment",
            CURATION: "Eval",
        }

        # Populate data
        data = []
        for sl in self.slices:
            slice_name = self.ident_mapping[sl.identifier]
            slice_size = len(sl)
            slice_category = category_names.get(sl.category, sl.category.capitalize())

            row = [slice_category, slice_name]

            for model in models:
                model_metrics = self.metrics[model]
                if sl.identifier not in model_metrics:
                    continue
                slice_metrics = model_metrics[sl.identifier]
                for agg in common_aggregators:
                    row.append(slice_metrics[agg])

            row.append(slice_size)
            data.append(row)

        df = pd.DataFrame(data)

        report = Report(
            data=df,
            columns=columns,
            dataset_name=str(self.dataset.identifier),
        )
        report.sort(
            category_order=dict(
                (cat, i)
                for i, cat in enumerate(
                    [SUBPOPULATION, AUGMENTATION, CURATION, ATTACK, GENERIC]
                )
            )
        )
        return report

    def search(self, keyword: str, limit: int = 3):
        """Fuzzy search over the slices in the DevBench."""
        return [
            self._slice_table[t[0]]
            for t in process.extract(keyword, self._slice_identifiers, limit=limit)
        ]

    def save(self, path: str) -> None:
        """Save the current devbench to disk. This will save all slices in the
        devbench to disk, as well as metrics and other metadata associated with
        this devbench.

        Args:
            path: string path to the save directory

        Returns: None
        """

        # Path to the save directory
        savedir = pathlib.Path(path) / f"{self.identifier}"

        # Create a directory inside savedir for the slices
        (savedir / "slices").mkdir(parents=True, exist_ok=True)

        # Save all the slices
        self._dataset.save_to_disk(str(savedir / "dataset"))
        pbar = tqdm(self.slices)
        for sl in pbar:
            pbar.set_description(f"Saving slice {str(sl.identifier)[:100]}...")
            sl.save_to_disk(
                str(savedir / "slices" / str(persistent_hash(str(sl.identifier))))
            )

        # Save metrics
        dill.dump(self.metrics, open(str(savedir / "metrics.dill"), "wb"))

        # Save aggregators
        dill.dump(self.aggregators, open(str(savedir / "aggregators.dill"), "wb"))

        # Save metadata
        dill.dump(
            {
                "identifier": self.identifier,
            },
            open(str(savedir / "metadata.dill"), "wb"),
        )

        # Save version info
        with open(str(savedir / "version.dill"), "wb") as f:
            f.write(self._dumps_version())

    @classmethod
    def available(cls, path: str) -> List[str]:
        """Check the list of available devbenches in a directory.

        Args:
            path: string path to a directory. The devbenches available inside this
            directory will be returned.

        Returns: list of available devbenches
        """

        # Path to the save directory
        savedir = pathlib.Path(path)

        # Loop over the folders
        devbench_identifiers = []
        for maybe_devbench in savedir.glob("*"):
            if maybe_devbench.is_dir() and (maybe_devbench / "metadata.dill").exists():
                devbench_identifiers.append(maybe_devbench.name)

        return devbench_identifiers

    @classmethod
    def load(cls, path: str) -> DevBench:
        """Load a devbench from disk.

        Args:
            path: string path to the devbench directory

        Returns:
        """

        # Path to the save directory
        savedir = pathlib.Path(path)

        # Load all the slices
        slices = []
        for sl_path in tqdm(list((savedir / "slices").glob("*"))):
            try:
                slices.append(Slice.load_from_disk(str(sl_path)))
            except FileNotFoundError:
                continue

        # Load dataset
        dataset = Dataset.load_from_disk(str(savedir / "dataset"))

        # Load metrics
        metrics = dill.load(open(str(savedir / "metrics.dill"), "rb"))

        # Load metrics
        aggregators = dill.load(open(str(savedir / "aggregators.dill"), "rb"))

        # Load metadata
        _ = dill.load(open(str(savedir / "metadata.dill"), "rb"))

        # Create the devbench
        devbench = cls(
            dataset=dataset,
        )

        # Set previously stored metrics
        devbench.metrics = metrics

        # Set previously stored aggregators
        devbench.aggregators = aggregators

        # Set the slices
        devbench.add_slices(slices)

        # Load version info
        with open(str(savedir / "version.dill"), "rb") as f:
            devbench._loads_version(f.read())

        return devbench


# TODO(karan): make the TestBench hashable
class TestBench(SemanticVersionerMixin):
    """Class for test benches in Robustness Gym."""

    def __init__(
        self,
        identifier: Union[str, Identifier],
        task: Task = None,
        slices: Collection[Union[Slice, Dataset]] = None,
        version: str = "0.0.1",
        dataset_id: str = None,
        class_names: Collection[str] = None,
    ):

        # Call the superclass
        super(TestBench, self).__init__(version=version)

        # An identifier for the TestBench
        self.identifier = identifier

        # Set the task
        self.task = task
        if task is None:
            self.task = Task()

        # Create the collection of slices
        self._slices = set()
        self.slice_identifiers = set()
        self._slice_table = {}

        # Add slices if any
        if slices:
            self.add_slices(slices)

        # The testbench has calculators
        self.calculators = set()

        # The testbench internally tracks metrics
        self.metrics = {}

        # The schema tells the testbench which columns to extract from the slices for
        # evaluation
        self.schema_type = "default"

        self.dataset_id = dataset_id

        self.class_names = class_names

    @property
    def slices(self):
        """Slices in the testbench."""
        return self._slices

    def __repr__(self):
        return f"TestBench[{self.identifier}](slices={len(self.slices)})"

    def _digest(self) -> str:
        return json.dumps([str(sl) for sl in self.slices])

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        input_columns: List[str],
        output_columns: List[str],
        # prediction_columns: List[str],
        # metrics: List[str],
    ) -> TestBench:
        """Create a TestBench from a dataset."""
        # Define the task
        task = Task(
            # Identifier
            Identifier("Task", dataset=str(dataset.identifier)),
            # Input and output schemas
            *Schema.for_dataset(dataset, input_columns, output_columns),
        )

        # Create the testbench
        testbench = TestBench(
            identifier=Identifier("TestBench", dataset=str(dataset.identifier)),
            task=task,
            slices=[dataset],
        )

        # testbench.set_single_dataset_mode()
        # testbench.set_prediction_columns(prediction_columns)

        return testbench

    def set_prediction_columns(self, prediction_columns: List[str]):
        """Set the list of columns that act as prediction columns."""
        self.prediction_columns = prediction_columns

    def set_single_dataset_mode(self):
        """All slices must be derived from the root dataset in single dataset
        mode."""
        self.single_dataset_mode = True

    def add_calculators(self, calculators: List[Union[str, Callable]]):
        """Add a list of calculators."""
        for calc in calculators:
            if isinstance(calc, str):
                self.calculators.add(get_metric(calc))
            elif isinstance(calc, Callable):
                self.calculators.add(calc)
            else:
                continue

    def add_model(
        self,
        model: Model = None,
        prediction_columns: List[str] = None,
        identifier: Union[str, Identifier] = None,
    ):

        if prediction_columns is not None:
            assert model is None, "`model` must be None."
            assert (
                identifier is not None
            ), "A model `identifier` must be included with `prediction_columns`."

            self.metrics[identifier] = {}
            for sl in self.slices:
                self.metrics[identifier][sl.identifier] = {}
                for calc in self.calculators:
                    self.metrics[identifier][sl.identifier][calc.__name__] = calc(
                        *[sl[col] for col in prediction_columns],
                        *[sl[col] for col in self.task.output_schema.columns],
                    )

    @classmethod
    def for_dataset(
        cls,
        dataset: str,
        task: Optional[Union[str, Task]] = None,
        version: str = None,
    ):
        """Create a test bench for a dataset."""

        inferred_task = None
        if task is not None:
            # Infer the task from the dataset
            inferred_task = Task.lookup(dataset=dataset)()
            # Check that the inferred task matches the task argument
            if task is not None and task != inferred_task:
                raise AssertionError(
                    f"Dataset {dataset} is only compatible with {inferred_task}, "
                    f"not {task}."
                )

        return TestBench(
            identifier=f"{dataset}-{task}-{version}",
            task=inferred_task,
            slices=[],
        )

    @classmethod
    def for_task(
        cls,
        task: Union[str, Task],
        version: str = None,
    ):
        """Create a testbench for a task."""
        return TestBench(
            identifier=f"{task}-{version}",
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
                group_info[key] = "name"
            else:
                only_single_column = len(set([t[1] for t in group])) == 1
                if only_single_column:
                    group_info[key] = "builder_ident"
                else:
                    group_info[key] = "full"

        ident_mapping = {}
        for ident in self.slice_identifiers:
            if "->" in str(ident):
                builder_ident = str(ident).split(" -> ")[-1]
                builder_ident, cols = builder_ident.split(" @ ")
                name = builder_ident.split("(")[0]

                if group_info[name] == "name":
                    new_ident = name
                elif group_info[name] == "builder_ident":
                    new_ident = builder_ident
                elif group_info[name] == "full":
                    new_ident = str(ident).split(" -> ")[-1]

                if new_ident.startswith("NlpAugTransformation"):
                    new_ident = new_ident.split("NlpAugTransformation(pipeline=[")[
                        1
                    ].split("])")[0]

            else:
                new_ident = str(ident).split("(")[0]

            ident_mapping[ident] = new_ident

        self.ident_mapping = ident_mapping

    def add_slices(self, slices: Collection[Union[Dataset, Slice]]) -> None:
        """Add slices to the testbench.

        Args:
            slices: collection of Slice objects
        """
        if isinstance(slices, Slice) or isinstance(slices, Dataset):
            slices = [slices]

        # Add slices
        for sl in slices:
            if not isinstance(sl, Slice) and isinstance(sl, Dataset):
                # The slice is a Dataset
                sl = Slice(sl)

            if isinstance(sl, Slice):
                if sl.identifier not in self.slice_identifiers and len(sl) > 0:
                    # Add slices that aren't already present in the testbench and have
                    # non-zero length
                    self.slices.add(sl)
                    self.slice_identifiers.add(sl.identifier)
                    self._slice_table[sl.identifier] = sl

    def evaluate(
        self,
        model: Model,
        batch_size: int = 32,
        coerce_fn: Callable = None,
        input_columns: List[str] = None,
        output_columns: List[str] = None,
    ) -> Dict:
        """Evaluate a model using the test bench and cache results.

        Args:
            model: model to evaluate
            batch_size: batch size for inference
            coerce_fn: function to coerce the model's outputs. Useful if the model's
            outputs cannot directly be compared to the targets.
            input_columns: columns for input schema. Required if task is None.
            output_columns: columns for output schema. Required if task is None.

        Returns: dict mapping slice identifiers to evaluation metrics.
        """

        if self.task is None:
            if input_columns is None or output_columns is None:
                raise ValueError(
                    "Input and output columns required when no task specified."
                )
        else:
            # Set the schema using the task
            # TODO Is the remapping required when not using a task
            self.set_schema("task")
            input_columns = self.task.input_schema.columns
            output_columns = self.task.output_schema.columns

        # TODO(karan): Uncomment and fix this assert on the type of outputs that
        #  model(..) returns
        # # Grab 2 examples from the first slice, run it through the model and check
        # that the output is a dictionary
        # output = model(dataset=Dataset.from_batch(self.slices[0][:2]),
        #                input_keys=self.task.input_schema.keys(),
        #                output_keys=self.task.output_schema.keys(),
        #                batch_size=2,
        #                coerce_fn=coerce_fn)
        # print(output)
        # assert isinstance(output, Sequence) and isinstance(output[0], Mapping), \
        #     "model(..) must return a list of dictionaries. Each dictionary should
        #     map metric names to values."

        # Store the model_metrics
        if model.identifier not in self.metrics:
            self.metrics[model.identifier] = {}

        # Run the model on all the slices
        # TODO(karan): For slices that are subpopulations, the same example can be in
        #  multiple slices
        #  and will be run through the model multiple times. Create a UnionSlice?
        for sl in tqdm(self.slices):
            if sl.identifier not in self.metrics[model.identifier]:
                # Evaluate on the slice
                # TODO Why not update existing results?
                self.metrics[model.identifier][sl.identifier] = model.evaluate(
                    dataset=sl,
                    input_columns=input_columns,
                    output_columns=output_columns,
                    batch_size=batch_size,
                    coerce_fn=coerce_fn,
                )

        return self.metrics[model.identifier]

    def add_predictions(
        self,
        model: Union[Model, str],
        predictions: Dict[str, Union[Sequence, torch.Tensor]],
        output_columns: List[str] = None,
        num_classes=None,
        metrics: List[str] = None,
    ) -> Dict:
        """Compute and cache metrics for pre-computed model predictions
        Args:
            model: Model or model id
            predictions: Map from slice id to sequence or torch Tensor of predictions
            metric (optional): list of metrics. If None, use the metrics specified in
            the task.
            output_columns (optional): names of output columns. Required if testbench
            does not have associated task.
            num_classes (optional): number of classes. Required if testbench does not
            have associated task.
        Returns: computed metrics
        """

        if self.task is None:
            if output_columns is None:
                raise ValueError(
                    "'output_columns' is required if testbench does not have "
                    "associated task."
                )
            if num_classes is None:
                raise ValueError(
                    "'num_classes' is required if testbench does not have associated "
                    "task."
                )
            if metrics is None:
                raise ValueError(
                    "'metrics' is required if testbench does not have associated task."
                )
        else:
            output_columns = self.task.output_schema.columns
            num_classes = self.task.output_schema.features[
                list(self.task.output_schema.columns)[0]
            ].num_classes
            if self.task.classification():
                assert len(output_columns) == 1  # , "Only supports classification."
            if metrics is None:
                metrics = self.task.metrics

        if len(output_columns) > 1:
            raise NotImplementedError("Only single output column supported")

        if isinstance(model, Model):
            model = model.identifier
        if model not in self.metrics:
            self.metrics[model] = {}
        for sl in tqdm(self.slices):
            if sl.identifier not in self.metrics[model]:
                # Evaluate on the slice
                # TODO Why not update existing results?
                # slice_predictions = predictions[sl.identifier]
                evaluation_dict = {}
                # Temporarily expose prediction columns
                # sl.set_format(columns=output_columns()
                # slice_predictions = predictions[sl.identifier]
                # TODO Optimize
                # labels = list(zip(*[sl[col] for col in output_columns]))
                labels = sl[output_columns[0]]
                for metric in metrics:
                    evaluation_dict[metric] = compute_metric(
                        metric=metric,
                        predictions=predictions[sl.identifier],
                        labels=labels,
                        num_classes=num_classes,
                    )
                # sl.reset_format()
                self.metrics[model][sl.identifier] = evaluation_dict

        return evaluation_dict

    def add_metrics(self, model: Union[Model, str], metrics: Dict[str, float]):
        """Cache pre-computed metrics for model
        Args:
            model: Model or model id.
            metrics: map from metric name to value
        """
        if isinstance(model, Model):
            model = model.identifier
        self.metrics[model] = metrics

    def create_report(
        self,
        model: Union[Model, str],
        metric_ids: List[str] = None,
    ) -> Report:
        """Generate report from cached metrics for a model
        Args:
            model: Model or model id. Metrics must have already been computed for
            this model.
            metric_ids (optional): list of metric ids to include in desired order.
            If None, take metrics from sample slice.
        Returns:
            report
        """

        if len(self.slices) == 0:
            raise ValueError("Cannot create report for empty testbench")

        if isinstance(model, Model):
            model = model.identifier
        if model not in self.metrics:
            raise ValueError(
                f"Metrics for model {model} have not been computed yet."
                f" You must first execute one of "
                "the following methods for this model: 'evaluate', "
                "'add_predictions', 'add_metrics'"
            )

        # TODO(Jesse): Need a category for test set

        model_metrics = self.metrics[model]

        # TODO(Jesse): where to put this? Should only need to be called once
        self._human_readable_identifiers()

        if metric_ids is None:
            sample_slice = list(self.slices)[0].identifier
            metric_ids = list(model_metrics[sample_slice].keys())
            sorted_metric_ids = sorted(
                [
                    metric_id
                    for metric_id in metric_ids
                    if metric_id not in ("class_dist", "pred_dist")
                ]
            )
            if "class_dist" in metric_ids:
                sorted_metric_ids.append("class_dist")
            if "pred_dist" in metric_ids:
                sorted_metric_ids.append("pred_dist")
            metric_ids = sorted_metric_ids

        # Populate columns
        columns = []
        for metric_id in metric_ids:
            if metric_id in ("class_dist", "pred_dist"):
                if self.task is None:
                    class_cds = None
                else:
                    class_names = self.task.output_schema.features[
                        list(self.task.output_schema.columns)[0]
                    ].names
                    class_cds = [name[0].upper() for name in class_names]
                columns.append(ClassDistributionColumn(metric_id, class_cds))
            else:
                columns.append(
                    ScoreColumn(metric_id, min_val=0, max_val=1, is_0_to_1=True)
                )
        columns.append(NumericColumn("Size"))

        category_names = {
            GENERIC: "Slice",
            SUBPOPULATION: "SubPop",
            ATTACK: "Attack",
            AUGMENTATION: "Augment",
            CURATION: "Eval",
        }

        # Populate data
        data = []
        for sl in self.slices:
            slice_name = self.ident_mapping[sl.identifier]
            slice_size = len(sl)
            slice_category = category_names.get(sl.category, sl.category.capitalize())
            row = []
            row.append(slice_category)
            row.append(slice_name)
            if sl.identifier not in model_metrics:
                raise ValueError(
                    f"Metrics for model {model} and slice {sl.identifier}"
                    f"have not yet been computed."
                )
            slice_metrics = model_metrics[sl.identifier]
            for metric_id in metric_ids:
                row.append(slice_metrics[metric_id])
            row.append(slice_size)
            data.append(row)

        # TODO(karan): generalize aggregation
        # slice_metrics = tz.merge_with(np.mean, slice_metrics)
        # Task-dependent model predictions
        # TODO(karan): e.g. average class distribution predicted, figure out how to
        #  put this in
        # Task-dependent sl information
        # TODO(karan): e.g. class distribution

        df = pd.DataFrame(data)

        report = Report(
            data=df, columns=columns, model_name=model, dataset_name=self.dataset_id
        )
        report.sort(
            category_order=dict(
                (cat, i)
                for i, cat in enumerate(
                    [SUBPOPULATION, AUGMENTATION, CURATION, ATTACK, GENERIC]
                )
            )
        )
        return report

    def set_schema(self, schema_type: str):
        assert schema_type in {"default", "task"}

        if self.schema_type == schema_type:
            return

        if schema_type == "task":
            self._slices = {self.task.remap_schema(sl) for sl in self.slices}
            self.schema_type = schema_type
        elif schema_type == "default":
            # TODO(karan): undo the schema standardization
            raise NotImplementedError

    def search(self, keyword: str, limit: int = 3):
        return [
            self._slice_table[t[0]]
            for t in process.extract(keyword, self.slice_identifiers, limit=limit)
        ]

    def save(self, path: str) -> None:
        """Save the current testbench to disk. This will save all slices in the
        testbench to disk, as well as metrics and other metadata associated
        with this testbench.

        Args:
            path: string path to the save directory

        Returns: None
        """

        # Path to the save directory
        savedir = pathlib.Path(path) / f"{self.identifier}"

        # Create a directory inside savedir for the slices
        (savedir / "slices").mkdir(parents=True, exist_ok=True)

        # Save all the slices
        pbar = tqdm(self.slices)
        for sl in pbar:
            pbar.set_description(f"Saving slice {str(sl.identifier)[:100]}...")
            sl.save_to_disk(
                str(savedir / "slices" / str(persistent_hash(str(sl.identifier))))
            )

        # Save metrics
        dill.dump(self.metrics, open(str(savedir / "metrics.dill"), "wb"))

        # Save metadata
        dill.dump(
            {
                "task": self.task,
                "identifier": self.identifier,
                "dataset_id": self.dataset_id,
            },
            open(str(savedir / "metadata.dill"), "wb"),
        )

        # Save version info
        with open(str(savedir / "version.dill"), "wb") as f:
            f.write(self._dumps_version())

    @classmethod
    def available(cls, path: str) -> List[str]:
        """Check the list of available testbenches in a directory.

        Args:
            path: string path to a directory. The testbenches available inside this
            directory will be returned.

        Returns: list of available testbenches
        """

        # Path to the save directory
        savedir = pathlib.Path(path)

        # Loop over the folders
        testbench_identifiers = []
        for maybe_testbench in savedir.glob("*"):
            if (
                maybe_testbench.is_dir()
                and (maybe_testbench / "metadata.dill").exists()
            ):
                testbench_identifiers.append(maybe_testbench.name)

        return testbench_identifiers

    @classmethod
    def load(cls, path: str) -> TestBench:
        """Load a testbench from disk.

        Args:
            path: string path to the testbench directory

        Returns:
        """

        # Path to the save directory
        savedir = pathlib.Path(path)

        # Load all the slices
        slices = []
        for sl_path in tqdm(list((savedir / "slices").glob("*"))):
            try:
                slices.append(Slice.load_from_disk(str(sl_path)))
            except FileNotFoundError:
                continue

        # Load metrics
        metrics = dill.load(open(str(savedir / "metrics.dill"), "rb"))

        # Load metadata
        metadata = dill.load(open(str(savedir / "metadata.dill"), "rb"))

        # Create the testbench
        testbench = cls(
            identifier=metadata["identifier"],
            task=metadata["task"],
            slices=slices,
        )

        # Set previously stored metrics
        testbench.metrics = metrics

        # Load version info
        with open(str(savedir / "version.dill"), "rb") as f:
            testbench._loads_version(f.read())

        return testbench

    def make(self, identifier: str):
        # Resolve the location of the TestBench

        # Pull the TestBench
        return self.pull(identifier)

    def pull(self, identifier: str):
        pass

    def publish(self):
        pass
