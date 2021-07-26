from __future__ import annotations

import inspect
import json
import logging
import pathlib
from typing import Callable, Collection, Dict, List, Union

import dill
import pandas as pd
from fuzzywuzzy import process
from tqdm import tqdm

from robustnessgym.core.constants import (
    ATTACK,
    AUGMENTATION,
    CURATION,
    GENERIC,
    SUBPOPULATION,
)
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.core.tools import persistent_hash
from robustnessgym.core.version import SemanticVersionerMixin
from robustnessgym.report.report import NumericColumn, Report, ReportColumn, ScoreColumn
from robustnessgym.slicebuilders.slicebuilder import SliceBuilder

logger = logging.getLogger(__name__)


class ReportableMixin:
    def _shared_aggregators(self, models: List[str]):
        """Find aggregators that are shared by multiple models in the Bench."""

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
        aggregator_columns: Dict[str, ReportColumn] = None,
    ) -> Report:
        """Generate a report for models in the bench.

        Args:
            models (List[str]): names of one or more models that are in the devbench.
            aggregator_columns (Dict[str, (ReportColumn, dict)]):
                dict mapping aggregator names to a tuple.

                The first entry of the tuple is the ReportColumn that should be
                used for visualization. The second entry is a dict of kwargs that
                will be passed to the ReportColumn using
                `ReportColumn.__init__(..., **kwargs)`.

                For instance,
                >>> devbench.create_report(
                >>>     models=['BERT'],
                >>>     aggregator_columns={
                >>>         'accuracy': (ScoreColumn, {'min_val': 0.3})
                >>>     }
                >>> )

                By default, aggregators will be displayed as a ScoreColumn
                with `min_val=0`, `max_val=1` and `is_0_to_1=True`.

        Returns:
            a Report, summarizing the performance of the models.
        """

        if len(self.slices) == 0:
            raise ValueError("No slices found in Bench. Cannot create report.")

        if models is not None:
            for model in models:
                assert model in self.metrics, f"Model {model} not found."
        else:
            # Use all the models that are available
            models = list(self.metrics.keys())

        # Set identifiers to be human readable
        self._human_readable_identifiers()

        # Get the list of aggregators that are shared by `models`
        shared_aggregators = list(self._shared_aggregators(models))

        # Populate columns
        columns = []
        for model in models:
            for aggregator in shared_aggregators:
                if aggregator_columns and aggregator in aggregator_columns:
                    column_type, column_kwargs = aggregator_columns[aggregator]
                else:
                    column_type = ScoreColumn
                    column_kwargs = dict(min_val=0, max_val=1, is_0_to_1=True)
                columns.append(column_type(f"{model}-{aggregator}", **column_kwargs))
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
                for agg in shared_aggregators:
                    row.append(slice_metrics[agg])

            row.append(slice_size)
            data.append(row)

        df = pd.DataFrame(data)

        report = Report(
            data=df,
            columns=columns,
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

    def _human_readable_identifiers(self):
        # Temporary function to generate human readable names
        groups = {}
        for ident in self._slice_identifiers:
            if "->" in str(ident):
                builder_ident = str(ident).split(" -> ")[-1]
                try:
                    builder_ident, cols = builder_ident.split(" @ ")
                except ValueError:
                    cols = ""
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
                try:
                    builder_ident, cols = builder_ident.split(" @ ")
                except ValueError:
                    cols = ""
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


class DevBench(SemanticVersionerMixin, ReportableMixin):
    def __init__(self):
        super(DevBench, self).__init__()

        # The collection of slices
        self._slices = set()
        self._slice_identifiers = set()
        self._slice_table = {}

        # The devbench has aggregators
        self.aggregators = {}

        # The devbench internally tracks metrics
        self.metrics = {}

    @property
    def slices(self):
        """Slices in the devbench."""
        return self._slices

    @property
    def models(self):
        """Models in the devbench."""
        return list(self.aggregators.keys())

    @property
    def summary(self):
        """Summary of the devbench."""
        return pd.DataFrame(self.metrics)

    def __repr__(self):
        return f"{self.__class__.__name__}(slices={len(self.slices)})"

    def __call__(self, slicebuilder: SliceBuilder, dp: DataPanel, columns: List[str]):
        return self.add_slices(slicebuilder(dp, columns)[0])

    def _digest(self) -> str:
        return json.dumps([str(sl) for sl in self.slices])

    def add_slices(
        self,
        slices: Union[DataPanel, Collection[DataPanel]],
        overwrite: bool = False,
    ) -> None:
        """Add slices to the DevBench.

        Args:
            slices (Union[DataPanel, Collection[DataPanel]]): a single DataPanel or
                a collection of DataPanels
            overwrite (bool): overwrite any slice in `slices` if it already exists
                in the DevBench
        """
        if isinstance(slices, DataPanel):
            slices = [slices]

        assert all(
            [isinstance(sl, DataPanel) for sl in slices]
        ), "All slices must be DataPanels."

        if not overwrite and any(
            [sl.identifier in self._slice_identifiers for sl in slices]
        ):
            logger.warning(
                "Some slices in `slices` already exist in the DevBench, "
                "pass `overwrite=True` to overwrite them."
            )

        for sl in slices:
            if overwrite or (
                sl.identifier not in self._slice_identifiers and len(sl) > 0
            ):
                # Add slices that aren't present and have non-zero length
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
        name to aggregation functions.

        Args:
            aggregators (Dict[str, Dict[str, Callable]]): a dict with keys that
                are model names (str), and values that are also dicts.

                The value dicts map aggregator names (str) to
                aggregation functions that produce a value. Typically, the aggregator
                name will be a metric (e.g. "accuracy") of some kind.

                The aggregation function must be a single argument Callable that takes
                as input a DataPanel, and outputs a metric value. As an example,

                >>> def accuracy(dp: DataPanel) -> float:
                >>>    return (dp['pred'] == dp['label']).mean()

                The output type of the aggregation function can be arbitrary,
                e.g. metrics that are np.ndarray or strings are also supported.
        """

        # `aggregators`: model_name -> dict
        for model, aggs in aggregators.items():
            # `aggs`: aggregator_name -> Callable
            for agg_name, agg in aggs.items():
                assert isinstance(
                    agg, Callable
                ), f"Aggregators must be functions, but {agg} is not."
                arguments = inspect.getfullargspec(agg).args
                assert (
                    len(arguments) == 1
                ), f"Aggregators must be single argument but {agg} is not."

                # Store the aggregator
                if model not in self.aggregators:
                    self.aggregators[model] = {}
                self.aggregators[model][agg_name] = agg

        # Calculate all metrics
        self.calculate()

    def calculate(self) -> None:
        """Calculate all metrics that haven't been calculated yet.

        The DevBench internally tracks metrics for all models and aggregators that
        were previously added, across all added slices.

        This `calculate` method is triggered automatically when a new slice
        or a new model and aggregator are added to the DevBench.
        """

        # `self.aggregators`: model_name -> aggregator_name
        for model in self.aggregators:

            # Add the model to the metrics dict
            if model not in self.metrics:
                self.metrics[model] = {}

            for sl in self.slices:
                # Add the slice to the model's metrics dict
                if str(sl.identifier) not in self.metrics[model]:
                    self.metrics[model][str(sl.identifier)] = {}

                for agg_name, aggregator in self.aggregators[model].items():
                    if agg_name not in self.metrics[model][str(sl.identifier)]:
                        # Calculate the aggregated value
                        self.metrics[model][str(sl.identifier)][agg_name] = aggregator(
                            sl
                        )

    def search(self, keyword: str, limit: int = 3) -> List[DataPanel]:
        """Fuzzy search to find a slice in the devbench. Returns slices that
        have identifiers that best match the searched keyword.

        Args:
            keyword (str): search phrase. The keyword will be used to match slice
                identifiers in the devbench.
            limit (int): number of results to return. Defaults to 3.

        Returns:
            A list of slice DataPanels.
        """
        return [
            self._slice_table[t[0]]
            for t in process.extract(keyword, self._slice_identifiers, limit=limit)
        ]

    def save(self, path: str) -> None:
        """Save the devbench to disk.

        This will save all slices in the devbench to disk, as well as
        metrics and aggregators associated with this devbench.

        Args:
            path: string path to the save directory e.g. "./my_analysis".
                A `.devbench` extension is added, so the devbench will be stored at
                "./my_analysis.devbench".

                To load the devbench back in, use `DevBench.load("./my_analysis")` or
                `DevBench.load("./my_analysis.devbench")`.
        """
        # Path to the save directory
        savedir = pathlib.Path(path)
        savedir = savedir.with_suffix(".devbench")

        # Create a directory inside savedir for the slices
        (savedir / "slices").mkdir(parents=True, exist_ok=True)

        # Save all the slices
        pbar = tqdm(self.slices)
        for sl in pbar:
            pbar.set_description(f"Saving slice {str(sl.identifier)[:100]}...")
            sl.write(str(savedir / "slices" / str(persistent_hash(str(sl.identifier)))))

        # Save metrics
        dill.dump(self.metrics, open(str(savedir / "metrics.dill"), "wb"))

        # Save aggregators
        dill.dump(self.aggregators, open(str(savedir / "aggregators.dill"), "wb"))

        # Save version info
        with open(str(savedir / "version.dill"), "wb") as f:
            f.write(self._dumps_version())

    @classmethod
    def available(cls, path: str) -> List[str]:
        """Check what devbenches are available in a directory.

        Args:
            path (str): directory path.
                The devbenches available inside this directory will be returned.

        Returns:
            names of available devbenches in a list.
        """

        # Path to the save directory
        savedir = pathlib.Path(path)

        devbenches = []
        for maybe_devbench in savedir.glob("*"):

            # DevBench is saved to a directory
            if maybe_devbench.is_dir():
                if (maybe_devbench / "metadata.dill").exists():  # TODO: deprecate
                    devbenches.append(maybe_devbench.name)
                elif maybe_devbench.suffix == ".devbench":
                    devbenches.append(maybe_devbench.name)
                else:
                    continue

        return devbenches

    @classmethod
    def load(cls, path: str) -> DevBench:
        """Load a devbench from disk.

        Args:
            path (str): path to the devbench directory. The devbench directory must
                have the `.devbench` extension.

        Returns:
            a DevBench
        """

        # Path to the save directory
        savedir = pathlib.Path(path)

        # Load all the slices
        slices = []
        for sl_path in tqdm(list((savedir / "slices").glob("*"))):
            try:
                slices.append(DataPanel.read(str(sl_path)))
            except FileNotFoundError:
                continue

        # Load metrics
        metrics = dill.load(open(str(savedir / "metrics.dill"), "rb"))

        # Load metrics
        aggregators = dill.load(open(str(savedir / "aggregators.dill"), "rb"))

        # Create the devbench
        devbench = cls()

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
