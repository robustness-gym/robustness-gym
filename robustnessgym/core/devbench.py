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
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.report import NumericColumn, Report, ScoreColumn
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.core.tools import persistent_hash
from robustnessgym.core.version import SemanticVersionerMixin

logger = logging.getLogger(__name__)


class DevBench(SemanticVersionerMixin):
    def __init__(
        self,
        dp: DataPanel,
    ):
        super(DevBench, self).__init__()

        # An identifier for the DevBench
        self.identifier = Identifier("DevBench", dataset=str(dp.identifier))

        # DataPanel that the devbench operates on
        self._dp = dp

        # The collection of slices
        self._slices = set()
        self._slice_identifiers = set()
        self._slice_table = {}

        # The devbench has aggregators
        self.aggregators = {}

        # The devbench internally tracks metrics
        self.metrics = {}

        # Add slices if any
        self.add_slices(dp)

    @property
    def datapanel(self):
        """DataPanel for the devbench."""
        return self._dp

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
        slices: Union[DataPanel, DataPanel, Collection[Union[DataPanel, DataPanel]]],
    ) -> None:
        """Add slices to the development bench.

        Args:
            slices: collection of DataPanel objects
        """
        if isinstance(slices, DataPanel) or isinstance(slices, DataPanel):
            slices = [slices]

        # Add slices
        for sl in slices:
            if not isinstance(sl, DataPanel) and isinstance(sl, DataPanel):
                # Convert `DataPanel` to `DataPanel`
                sl = DataPanel(sl)

            if isinstance(sl, DataPanel):
                if (
                    sl.identifier not in self._slice_identifiers
                    and len(sl) > 0
                    and sl.lineage[0][1] == self.datapanel.identifier
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
        dataset: DataPanel,
    ) -> DevBench:
        """Create a DevBench from a dataset."""
        # Create the devbench
        devbench = DevBench(
            dp=dataset,
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
            dataset_name=str(self.datapanel.identifier),
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
        self._dp.save_to_disk(str(savedir / "dataset"))
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
            {"identifier": self.identifier},
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
        # TODO: fix DataPanel load from disk

        # Load all the slices
        slices = []
        for sl_path in tqdm(list((savedir / "slices").glob("*"))):
            try:
                slices.append(DataPanel.read(str(sl_path)))
            except FileNotFoundError:
                continue

        # Load dataset
        dp = DataPanel.read(str(savedir / "dataset"))

        # Load metrics
        metrics = dill.load(open(str(savedir / "metrics.dill"), "rb"))

        # Load metrics
        aggregators = dill.load(open(str(savedir / "aggregators.dill"), "rb"))

        # Load metadata
        _ = dill.load(open(str(savedir / "metadata.dill"), "rb"))

        # Create the devbench
        devbench = cls(
            dp=dp,
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
