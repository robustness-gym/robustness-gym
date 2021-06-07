from __future__ import annotations

import itertools
from functools import partial
from typing import Dict, List

import dill
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots


class ReportColumn:
    """A single column in the Robustness Report."""

    def __init__(self, title: str):
        self.title = title

    def set_title(self, title: str):
        self.title = title


class ScoreColumn(ReportColumn):
    """A column for numeric scores in the Robustness Report, displayed as a bar
    chart."""

    def __init__(
        self, title: str, min_val: float, max_val: float, is_0_to_1: bool = False
    ):
        super(ScoreColumn, self).__init__(title)
        self.min_val = min_val
        self.max_val = max_val
        self.is_0_to_1 = is_0_to_1

    def set_min(self, min_val: float):
        self.min_val = min_val

    def set_max(self, max_val: float):
        self.max_val = max_val


class ClassDistributionColumn(ReportColumn):
    """A column for discrete class distributions in the Robustness Report,
    displayed as a heatmap."""

    def __init__(self, title: str, class_codes: List[str]):
        super(ClassDistributionColumn, self).__init__(title)
        self.class_codes = class_codes

    def set_class_codes(self, class_codes: List[str]):
        self.class_codes = class_codes


class NumericColumn(ReportColumn):
    """A column for numeric data in the Robustness Report, displayed as the raw
    value."""

    def __init__(self, title: str):
        super(NumericColumn, self).__init__(title)


class Report:
    """Class for Robustness Gym Report.
    Args:
        data: Pandas dataframe in the following format:
            column 1: category name
            column 2: slice name
            columns 3-N: data corresponding to passed columns parameter
        columns: ReportColumn objects specifying format of columns 3-N in data
        model_name (optional): model name to show in report
        dataset_name (optional): dataset name to show in report
        **kwargs (optional): any additional config paramters
    """

    def __init__(
        self,
        data: pd.DataFrame,
        columns: List[ReportColumn],
        model_name: str = None,
        dataset_name: str = None,
        **kwargs,
    ):

        # Make a copy of data since may be modified by methods below
        self.data = data.copy()

        self.columns = columns
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.config = {
            "color_scheme": ["#ec7734", "#3499ec", "#ec34c1", "#9cec34"],
            "score_color_complement": "#F3F4F7",
            "text_fill_color": "#F3F4F7",
            "text_border_color": "#BEC4CE",
            "distribution_color_scale": [[0.0, "#FBF5F2"], [1.0, "#EC7734"]],
            "col_spacing": 0.035,
            "row_height": 24,
            "category_padding": 24,
            "header_padding": 80,
            "score_col_width": 0.6,
            "class_dist_col_width": 0.35,
            "numeric_col_width": 0.25,
            "layout_width": 960,
            "font_size_dist": 12,
            "font_size_data": 13,
            "font_size_heading": 14,
            "font_size_category": 14,
        }

        self.update_config(**kwargs)

    def sort(
        self, category_order: Dict[str, int] = None, slice_order: Dict[str, int] = None
    ):
        """Sort rows in report by category / slice alphabetically, or using
        specified order.

        Args:
          category_order (optional): map from category name to sorting rank. If None,
          sort categories alphabetically.
          slice_order (optional): map from slice name to sorting rank. If None, sort
          slices alphabetically (within a category).
        """

        if category_order is None:
            category_order = {}

        if slice_order is None:
            slice_order = {}

        for col_name in ["sort-order-category", "sort-order-slice"]:
            if col_name in self.data:
                raise ValueError(f"Column name '{col_name}' is reserved")

        self.data["sort-order-category"] = self.data[0].map(
            lambda x: (category_order.get(x, 2 ** 10000), x)
        )
        self.data["sort-order-slice"] = self.data[1].map(
            lambda x: (slice_order.get(x, 2 ** 10000), x)
        )

        self.data = self.data.sort_values(
            by=["sort-order-category", "sort-order-slice"]
        ).drop(["sort-order-category", "sort-order-slice"], axis="columns")

        self.data.reset_index(inplace=True, drop=True)

    def filter(self, categories: List[str] = None, slices: List[str] = None):
        """Filter report to specific categories AND slices
        Args:
          categories (optional): list of category names to filter by
          slices (optional):list of slice names to filter by
        """
        if categories is not None:
            # self.data = self.data.loc(self.data[0].isin(categories))
            self.data = self.data[self.data[0].isin(categories)]
        if slices is not None:
            self.data = self.data[self.data[1].isin(slices)]
        self.data.reset_index(inplace=True, drop=True)

    def rename(self, category_map: Dict[str, str], slice_map: Dict[str, str]):
        """Rename categories, slices
        Args:
            category_map (optional): map from old to new category name
            slice_map (optional): map from old to new slice name
        """
        if category_map is not None:
            self.data[0] = self.data[0].map(lambda x: category_map.get(x, x))
        if slice_map is not None:
            self.data[1] = self.data[1].map(lambda x: slice_map.get(x, x))

    def set_class_codes(self, class_cds: List[str]):
        """Set single-letter class codes used for class distribution
        columns."""
        for col in self.columns:
            if isinstance(col, ClassDistributionColumn):
                col.set_class_codes(class_cds)

    def set_model_name(self, model_name):
        """Set model name displayed on report."""
        self.model_name = model_name

    def set_dataset_name(self, dataset_name):
        """Set dataset name displayed on report."""
        self.dataset_name = dataset_name

    def set_range(self, col_title: str, min_val: float = None, max_val: float = None):
        """Set min and max values for score columns
        Args:
            col_title: title of column to update
            min_val: minimum value
            max_val: maximum value
        """
        for col in self.columns:
            if isinstance(col, ScoreColumn) and col.title == col_title:
                if min_val is not None:
                    col.min_val = min_val
                if max_val is not None:
                    col.max_val = max_val

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.config:
                raise ValueError(f"Invalid config param: '{k}'")
            self.config[k] = v

    def round(self):
        # Round everything
        self.data = self.data.round(3)
        self.data.class_dist = self.data.class_dist.apply(partial(np.round, decimals=3))
        self.data.pred_dist = self.data.pred_dist.apply(partial(np.round, decimals=3))

    @classmethod
    def load(cls, path: str) -> Report:
        obj = dill.load(open(path, "rb"))
        assert isinstance(obj, Report), (
            f"dill loaded an instance of {type(obj)}, " f"must load {cls.__name__}."
        )
        return obj

    def save(self, path: str):
        return dill.dump(self, open(path, "wb"))

    def figure(self, show_title=False) -> Figure:

        # Verify that rows are grouped by category
        row_categories = self.data[0].tolist()
        save_cat_groups = set()  # Previous category groupings already encountered
        prev_cat = None
        # Loop through each row and see if a category is encountered outside of first
        # identified group for that category
        for cat in row_categories:
            if cat != prev_cat:  # category changes
                if cat in save_cat_groups:  # if new category previously encountered
                    raise ValueError("Rows must be grouped by category.")
                prev_cat = cat
                save_cat_groups.add(cat)

        categories = []
        category_sizes = []  # Num rows in each category
        for category, group in itertools.groupby(self.data[0]):  # column 0 is category
            categories.append(category)
            category_sizes.append(len(list(group)))
        n_rows = sum(category_sizes)
        height = (
            n_rows * self.config["row_height"]
            + len(categories) * self.config["category_padding"]
            + self.config["header_padding"]
        )
        col_widths = []
        for col in self.columns:
            if isinstance(col, ScoreColumn):
                col_width = self.config["score_col_width"]
            elif isinstance(col, ClassDistributionColumn):
                col_width = self.config["class_dist_col_width"]
            else:
                col_width = self.config["numeric_col_width"]
            col_widths.append(col_width)

        fig = make_subplots(
            rows=len(categories),
            row_titles=categories,
            cols=len(self.columns),
            shared_yaxes=True,
            subplot_titles=[col.title for col in self.columns],
            horizontal_spacing=self.config["col_spacing"],
            vertical_spacing=self.config["category_padding"] / height,
            row_width=list(reversed(category_sizes)),
            column_width=col_widths,
        )

        hms = []
        coords = []
        category_ndx = 1
        # Group data by category
        for category, category_data in self.data.groupby(0, sort=False):
            score_col_ndx = 0
            slice_names = category_data[1]
            slice_names = [s + " " * 3 for s in slice_names]
            for col_ndx, col in enumerate(self.columns):
                df_col_ndx = col_ndx + 2
                # Dataframe has two leading columns with category, slice
                fig_col_ndx = col_ndx + 1  # figure columns are 1-indexed
                x = category_data[df_col_ndx].tolist()
                if isinstance(col, ScoreColumn):
                    if col.is_0_to_1:
                        x = [100 * x_i for x_i in x]
                    col_max = col.max_val
                    if col.is_0_to_1:
                        col_max = 100 * col.max_val
                    fig.add_trace(
                        go.Bar(
                            x=x,
                            y=slice_names,
                            orientation="h",
                            marker=dict(color=self.get_color(score_col_ndx)),
                            showlegend=False,
                            text=[f"{x_i:.1f}" for x_i in x],
                            textposition="inside",
                            width=0.95,
                            textfont=dict(color="white"),
                        ),
                        row=category_ndx,
                        col=fig_col_ndx,
                    )
                    # Add marker for gray fill
                    fig.add_trace(
                        go.Bar(
                            x=[col_max - x_i for x_i in x],
                            y=slice_names,
                            orientation="h",
                            marker=dict(color=self.config["score_color_complement"]),
                            showlegend=False,
                            width=0.9,
                        ),
                        row=category_ndx,
                        col=fig_col_ndx,
                    )
                    score_col_ndx += 1
                elif isinstance(col, ClassDistributionColumn):
                    annotation_text = [
                        [f"{int(round(z * 100)):d}" for z in rw] for rw in x
                    ]
                    hm = ff.create_annotated_heatmap(
                        x,
                        x=col.class_codes,
                        xgap=1,
                        ygap=1,
                        annotation_text=annotation_text,
                        colorscale=self.config["distribution_color_scale"],
                        zmin=0,
                        zmax=1,
                    )
                    hms.append(hm)
                    # Save annotation data for special code related to heatmaps at end
                    coords.append(len(self.columns) * (category_ndx - 1) + fig_col_ndx)
                    fig.add_trace(
                        hm.data[0],
                        row=category_ndx,
                        col=fig_col_ndx,
                    )
                elif isinstance(col, NumericColumn):
                    # Repurpose bar chart as text field.
                    fig.add_trace(
                        go.Bar(
                            x=[1] * len(x),
                            y=slice_names,
                            orientation="h",
                            marker=dict(
                                color=self.config["text_fill_color"],
                                line=dict(
                                    width=0, color=self.config["text_border_color"]
                                ),
                            ),
                            showlegend=False,
                            text=[human_format(x_i) for x_i in x],
                            textposition="inside",
                            insidetextanchor="middle",
                            width=0.9,
                        ),
                        row=category_ndx,
                        col=fig_col_ndx,
                    )
                else:
                    raise ValueError("Invalid col type")
            category_ndx += 1

        for category_ndx in range(1, len(categories) + 1):
            if category_ndx == len(categories):
                show_x_axis = True
            else:
                show_x_axis = False
            for col_ndx, col in enumerate(self.columns):
                fig_col_ndx = col_ndx + 1  # plotly cols are 1-indexed
                fig.update_yaxes(autorange="reversed", automargin=True)
                if isinstance(col, ScoreColumn):
                    if col.is_0_to_1:
                        col_min, col_max = 100 * col.min_val, 100 * col.max_val
                    else:
                        col_min, col_max = col.min_val, col.max_val

                    fig.update_xaxes(
                        range=[col_min, col_max],
                        row=category_ndx,
                        col=fig_col_ndx,
                        tickvals=[col_min, col_max],
                        showticklabels=show_x_axis,
                    )
                elif isinstance(col, ClassDistributionColumn):
                    fig.update_xaxes(
                        row=category_ndx, col=fig_col_ndx, showticklabels=show_x_axis
                    )
                elif isinstance(col, NumericColumn):
                    fig.update_xaxes(
                        range=[0, 1],
                        row=category_ndx,
                        col=fig_col_ndx,
                        showticklabels=False,
                    )

        fig.update_layout(
            height=height,
            width=self.config["layout_width"],
            barmode="stack",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=self.config["font_size_data"]),
            yaxis={"autorange": "reversed"},
            margin=go.layout.Margin(
                r=0, b=0, t=20  # right margin  # bottom margin  # top margin
            ),
        )

        # Use low-level plotly interface to update padding / font size
        for a in fig["layout"]["annotations"]:
            # If label for group
            if a["text"] in categories:
                a["x"] = 0.99  # Add padding
                a["font"] = dict(size=self.config["font_size_category"])
            else:
                a["font"] = dict(
                    size=self.config["font_size_heading"]
                )  # Adjust font size for non-category labels

        # Due to a quirk in plotly, need to do some special low-level coding
        # Code from https://community.plotly.com/t/how-to-create-annotated-heatmaps
        # -in-subplots/36686/25
        newfont = [
            go.layout.Annotation(font_size=self.config["font_size_heading"])
        ] * len(fig.layout.annotations)
        fig_annots = [newfont] + [hm.layout.annotations for hm in hms]
        for col_ndx in range(1, len(fig_annots)):
            for k in range(len(fig_annots[col_ndx])):
                coord = coords[col_ndx - 1]
                fig_annots[col_ndx][k]["xref"] = f"x{coord}"
                fig_annots[col_ndx][k]["yref"] = f"y{coord}"
                fig_annots[col_ndx][k]["font_size"] = self.config["font_size_dist"]

        def recursive_extend(mylist, nr):
            # mylist is a list of lists
            result = []
            if nr == 1:
                result.extend(mylist[nr - 1])
            else:
                result.extend(mylist[nr - 1])
                result.extend(recursive_extend(mylist, nr - 1))
            return result

        new_annotations = recursive_extend(fig_annots[::-1], len(fig_annots))
        fig.update_layout(annotations=new_annotations)

        if show_title:
            title = {
                "text": f"{self.dataset_name or ''} {self.model_name or ''} "
                f"Robustness Report",
                "x": 0.5,
                "xanchor": "center",
            }
        else:
            title = None
        fig.update_layout(
            title=title,
            margin=go.layout.Margin(
                r=0, b=0, t=80  # right margin  # bottom margin  # top margin
            ),
        )

        return fig

    def get_color(self, col_ndx):
        return self.config["color_scheme"][col_ndx % len(self.config["color_scheme"])]


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )
