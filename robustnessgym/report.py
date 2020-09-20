import functools
import itertools
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ReportColumn:
    def __init__(self, name: str):
        self.name = name


class ScoreColumn(ReportColumn):
    def __init__(self, name: str, min: float, max: float):
        super(ScoreColumn, self).__init__(name)
        self.min = min
        self.max = max


class ClassDistributionColumn(ReportColumn):
    def __init__(self, name: str, class_codes: List[str]):
        super(ClassDistributionColumn, self).__init__(name)
        self.class_codes = class_codes


class NumericColumn(ReportColumn):
    def __init__(self, name: str):
        super(NumericColumn, self).__init__(name)


class Report:

    def __init__(self, data, columns, model_name, dataset_name):
        self.data = data
        self.columns = columns
        self.model_name = model_name
        self.dataset_name = dataset_name

    def figures(self, show_title=False):

        score_colors = [['#3499EC'],
                        ['#EC7734', '#3499EC'],
                        ['#EC7734', '#3499EC', '#ec34c1']]
        score_color_complement = '#F3F4F7'
        text_fill_color = '#F3F4F7'
        text_border_color = '#BEC4CE'
        distribution_color_scale = [[0.0, '#FBF5F2'], [1.0, '#EC7734']]
        col_spacing = 0.035

        categories = []
        category_sizes = []  # Num rows in each category
        for category, group in itertools.groupby(self.data['category']):
            categories.append(category)
            category_sizes.append(len(list(group)))
        row_height = 24
        category_padding = 24
        header_padding = 80
        n_rows = sum(category_sizes)
        height = n_rows * row_height + len(categories) * category_padding + header_padding
        col_widths = []
        for col in self.columns:
            if isinstance(col, ScoreColumn):
                col_width = 0.6
            elif isinstance(col, ClassDistributionColumn):
                col_width = 0.35
            else:
                col_width = 0.25
            col_widths.append(col_width)

        fig_detail = make_subplots(rows=len(categories),
                                   row_titles=categories,
                                   cols=len(self.columns),
                                   shared_yaxes=True,
                                   subplot_titles=[col.name.replace('_', ' ').title() for col in self.columns],
                                   horizontal_spacing=col_spacing,
                                   vertical_spacing=category_padding / height,
                                   row_width=list(reversed(category_sizes)),
                                   column_width=col_widths
                                   )

        summary_data = defaultdict(dict)
        score_cols = [col for col in self.columns if isinstance(col, ScoreColumn)]
        # n_scores = len(score_cols)

        hms = []
        coords = []
        category_ndx = 1
        for category, category_data in self.data.groupby('category'):
            score_col_ndx = 0
            slice_names = [s + '   ' for s in category_data['slice_name']]
            for col_ndx, col in enumerate(self.columns, start=1):
                x = category_data[col.name].tolist()
                if isinstance(col, ScoreColumn):
                    # HACK
                    if col.name.lower() in ('f1', 'accuracy', 'precision', 'recall'):
                        x = [100 * x_i for x_i in x]
                    fig_detail.add_trace(
                        go.Bar(
                            x=x,
                            y=slice_names,
                            orientation='h',
                            marker=dict(color=score_colors[len(score_cols) - 1][score_col_ndx]),
                            showlegend=False,
                            text=[f'{x_i:.1f}' for x_i in x],
                            textposition='inside',
                            width=.95,
                            textfont=dict(color='white')
                        ),
                        row=category_ndx, col=col_ndx
                    )
                    # Add marker for gray fill
                    fig_detail.add_trace(
                        go.Bar(
                            x=[col.max - x_i for x_i in x],
                            y=slice_names,
                            orientation='h',
                            marker=dict(color=score_color_complement),
                            showlegend=False,
                            width=.9
                        ),
                        row=category_ndx, col=col_ndx
                    )
                    # Accumulate summary statistics
                    summary_data[col.name][category] = min(x)
                    score_col_ndx += 1
                elif isinstance(col, ClassDistributionColumn):
                    annotation_text = [[f"{int(round(z * 100)):d}" for z in rw] for rw in x]
                    hm = ff.create_annotated_heatmap(x, x=col.class_codes, xgap=1, ygap=1,
                                                     annotation_text=annotation_text,
                                                     colorscale=distribution_color_scale,
                                                     zmin=0, zmax=1)
                    hms.append(hm)
                    # Save annotation data for special code related to heatmaps at end
                    coords.append(len(self.columns) * (category_ndx - 1) + col_ndx)
                    fig_detail.add_trace(
                        hm.data[0],
                        row=category_ndx, col=col_ndx,
                    )
                elif isinstance(col, NumericColumn):
                    # Repurpose bar chart as text field.
                    fig_detail.add_trace(
                        go.Bar(
                            x=[1] * len(x),
                            y=slice_names,
                            orientation='h',
                            marker=dict(color=text_fill_color,
                                        line=dict(width=0, color=text_border_color)),
                            showlegend=False,
                            text=[human_format(x_i) for x_i in x],
                            textposition='inside',
                            insidetextanchor='middle',
                            width=.9
                        ),
                        row=category_ndx, col=col_ndx
                    )
                else:
                    raise ValueError('Invalid col type')
            category_ndx += 1

        for category_ndx in range(1, len(categories) + 1):
            if category_ndx == len(categories):
                show_x_axis = True
            else:
                show_x_axis = False
            for col_ndx, col in enumerate(self.columns, start=1):
                fig_detail.update_yaxes(autorange='reversed', automargin=True)
                if isinstance(col, ScoreColumn):
                    fig_detail.update_xaxes(range=[col.min, col.max], row=category_ndx, col=col_ndx,
                                            tickvals=[col.min, col.max], showticklabels=show_x_axis)
                elif isinstance(col, ClassDistributionColumn):
                    fig_detail.update_xaxes(row=category_ndx, col=col_ndx, showticklabels=show_x_axis)
                elif isinstance(col, NumericColumn):
                    fig_detail.update_xaxes(range=[0, 1], row=category_ndx, col=col_ndx, showticklabels=False)

        fig_detail.update_layout(height=height,
                                 width=960,
                                 barmode='stack',
                                 plot_bgcolor='rgba(0, 0, 0, 0)',
                                 paper_bgcolor='rgba(0, 0, 0, 0)',
                                 font=dict(size=13),
                                 yaxis={'autorange': 'reversed'},
                                 margin=go.layout.Margin(
                                     # l=0,  # left margin
                                     r=0,  # right margin
                                     b=0,  # bottom margin
                                     t=20  # top margin
                                 )
                                 )

        # Use low-level plotly interface to update padding / font size
        for a in fig_detail['layout']['annotations']:
            # If label for group
            if a['text'] in categories:
                a['x'] = .99  # Add padding
            else:
                a['font'] = dict(size=14)  # Adjust font size for non-category labels

        # Due to a bug in plotly, need to do some special low-level coding
        # Code from https://community.plotly.com/t/how-to-create-annotated-heatmaps-in-subplots/36686/25
        newfont = [go.layout.Annotation(font_size=14)] * len(fig_detail.layout.annotations)
        fig_annots = [newfont] + [hm.layout.annotations for hm in hms]
        for col_ndx in range(1, len(fig_annots)):
            for k in range(len(fig_annots[col_ndx])):
                coord = coords[col_ndx - 1]
                fig_annots[col_ndx][k]['xref'] = f'x{coord}'
                fig_annots[col_ndx][k]['yref'] = f'y{coord}'
                fig_annots[col_ndx][k]['font_size'] = 11

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
        fig_detail.update_layout(annotations=new_annotations)

        # Generate summary figure
        if len(categories) >= 3:
            fig_summary = make_subplots(rows=1,
                                        cols=len(score_cols),
                                        # subplot_titles=score_names,
                                        specs=[[{'type': 'polar'}] * len(score_cols)],
                                        horizontal_spacing=.2,
                                        # column_width=[.35] * n_cols,
                                        )
            for i, col in enumerate(score_cols):
                # TODO Convention for the baseline blank category
                include_categories = [category for category in categories if category != '']
                category_scores = [summary_data[col.name][category] for category in include_categories]
                fig_summary.add_trace(go.Scatterpolar(
                    name=col.name,
                    r=category_scores,
                    theta=include_categories,
                    line=go.scatterpolar.Line(color=score_colors[len(score_cols) - 1][i])
                ), 1, i + 1)
            fig_summary.update_traces(fill='toself')
            if show_title:
                title = {'text': f"{self.dataset_name} {self.model_name} Robustness Report",
                         'y': .98,
                         'x': 0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'}
            else:
                title = None
            fig_summary.update_layout(height=330,
                                      width=960,
                                      margin=go.layout.Margin(
                                          # l=0,  # left margin
                                          # r=0,  # right margin
                                          b=60,  # bottom margin
                                          t=60  # top margin
                                      ),
                                      polar=dict(
                                          radialaxis=dict(
                                              visible=True,
                                              range=[col.min, col.max]
                                          )
                                      ),
                                      polar2=dict(
                                          radialaxis=dict(
                                              visible=True,
                                              range=[col.min, col.max]
                                          )
                                      ),
                                      polar3=dict(
                                          radialaxis=dict(
                                              visible=True,
                                              range=[col.min, col.max]
                                          )
                                      ),
                                      title=title
                                      )

            fig_summary.update_yaxes(automargin=True)
        else:
            fig_summary = None
            if show_title:
                title = {
                    'text': f"{self.dataset_name} {self.model_name} Robustness Report",
                    # 'y': .98,
                    'x': 0.5,
                    'xanchor': 'center', }
                # 'yanchor': 'top'
            else:
                title = None
            fig_detail.update_layout(
                title=title,
                margin=go.layout.Margin(
                    # l=0,  # left margin
                    r=0,  # right margin
                    b=0,  # bottom margin
                    t=80  # top margin
                )
            )

        self._figures = fig_summary, fig_detail
        return self._figures

    def latex(self):
        conf_to_latex = {}
        refs = "TextAttack (Morris 2020), TextFooler (Jin 2019))"
        conf_to_latex['iclr2021'] = f"""\
\\section{{Appendix}}
\\begin{{figure}}[h]
\\begin{{center}}
    \\includegraphics[width=\\linewidth]{{../images/robustness_gym_detail.pdf}}
\\end{{center}}
\\caption{{Robustness report for {self.model_name} model on {self.dataset_name} dataset. Summary view (top) shows the minimum score within
 each robustness category for each metric. Detail view (bottom) reports score for each robustness test, broken out by
 category. Tests include: {refs}}}.
 \end{{figure}}"""
        return conf_to_latex

    def write_appendix(self, outdir):
        reports_path = Path(__file__).parent.parent / 'reports'
        shutil.copytree(reports_path / 'template', outdir)
        image_path = outdir / 'images'
        image_path.mkdir(parents=True, exist_ok=True)
        fig_summary, fig_detail = self.figures(show_title=True)
        if fig_summary is not None:
            fig_summary.write_image(str(image_path / 'robustness_gym_summary.pdf'))
        fig_detail.write_image(str(image_path / 'robustness_gym_detail.pdf'))
        for conf_id, latex in self.latex().items():
            with open(outdir / conf_id / 'robustness_gym_appendix.tex', 'w') as f:
                f.write(latex)


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


if __name__ == "__main__":
    # import streamlit as st
    test_data = False
    if test_data:
        data = [
            # First group
            {'group': '',  # Name of group
             'slice_name': ['Test split'],  # Tests/slices for group
             'data': {  # Data, indexed by column name
                 'Accuracy': [55],
                 'F1': [45],
                 'Precision': [45],
                 'Class %': [
                     [0.3, 0.4, 0.3]
                 ],
                 'Pred. Class %': [
                     [0.27, 0.43, 0.3]
                 ],
                 'Size': [5000]
             }
             },
            # Second group, etc.
            {'group': 'slice_name',
             'slice_name': ['Negation', 'Contains -ing', 'Temporal preposition', 'Ends with verb', 'slice 5',
                            'slice 6', 'slice 7'],
             'data': {
                 'Accuracy': [50, 40, 30, 20, 10, 30, 20],
                 'F1': [20, 30, 10, 75, 80, 75, 80],
                 'Precision': [20, 30, 10, 75, 80, 75, 80],
                 'Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.2, 0.4, 0.4],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6]
                 ],
                 'Pred. Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.3, 0.4, 0.4],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                 ],
                 'Size': [15216, 18812, 123, 4321, 1234567, 40000, 15216],
             }
             },
            {'group': 'Augmentations',
             'slice_name': ['Augmentation 1', 'Augmentation 2', 'Augmentation 3',
                            'Augmentation 4', 'Augmentation 5', 'Augmentation 6', 'Augmentation 7'],
             'data': {
                 'Accuracy': [50, 40, 30, 20, 50, 20, 40],
                 'F1': [50, 50, 60, 75, 80, 75, 80],
                 'Precision': [50, 50, 60, 75, 80, 75, 80],
                 'Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.1, 0.6, 0.3],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6]
                 ],
                 'Pred. Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.2, 0.5, 0.3],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6]
                 ],
                 'Size': [15216, 18812, 123, 4321, 1234567, 40000, 15216],
             }
             },
            {'group': 'TextAttack',
             'slice_name': ['Textfooler', 'Hotflip', 'Morpheus', 'Seq2Sick', 'Hotflip 2',
                            'Morpheus 2', 'Seq2Sick 2'],
             'data': {
                 'Accuracy': [50, 40, 30, 40, 40, 50, 70],
                 'F1': [60, 50, 40, 75, 80, 40, 75, 80],
                 'Precision': [60, 50, 40, 75, 80, 40, 75, 80],
                 'Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.8, 0.1, 0.1],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                 ],
                 'Pred. Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.3, 0.4, 0.2],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.1, 0.8],
                     [0.1, 0.3, 0.6],
                 ],
                 'Size': [15216, 18812, 123, 4321, 1234567, 40000, 15216],
             }
             },
            {'group': 'Eval Sets',
             'slice_name': ['Eval set 1', 'Eval set 2', 'Eval set 3', 'Eval set 4', 'Eval set 5',
                            'Eval set 6', 'Eval set 7'],
             'data': {
                 'Accuracy': [50, 40, 30, 20, 10, 20, 10],
                 'F1': [20, 30, 10, 75, 80, 10, 75, 80],
                 'Precision': [20, 30, 10, 75, 80, 10, 75, 80],
                 'Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.2, 0.6, 0.2],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                 ],
                 'Pred. Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.4, 0.4, 0.2],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                 ],
                 'Size': [15216, 18812, 123, 4321, 1234567, 40000, 15216],
             }
             }
        ]

        # cols = [
        #     {'type': 'score', 'name': 'Accuracy', 'min': 0, 'max': 100},
        #     {'type': 'distribution', 'name': 'Class %', 'class_codes': ['E', 'N', 'C']},
        #     {'type': 'distribution', 'name': 'Pred. Class %', 'class_codes': ['E', 'N', 'C']},
        #     {'type': 'text', 'name': 'Size'},
        # ]

        cols = [
            ScoreColumn('Accuracy', 0.0, 100.0),
            ClassDistributionColumn('Class %', ['E', 'N', 'C']),
            ClassDistributionColumn('Pred. Class %', ['E', 'N', 'C']),
            NumericColumn('Size')
        ]

        col_values = defaultdict(list)
        for row_ndx, row in enumerate(data):
            group_name = row['group']
            num_slices = None
            for col_ndx, col in enumerate(cols):
                x = row['data'][col.name]
                col_values[col.name].extend(x)
                if num_slices is None:
                    num_slices = len(x)
                else:
                    assert num_slices == len(x)
            col_values['category'].extend([group_name] * num_slices)
            col_values['slice_name'].extend(row['slice_name'])

        df = pd.DataFrame(col_values,
                          columns=['category'] + ['slice_name'] + [col.name for col in cols])
        print(df)
        report = Report(df, cols, 'SNLI', 'BERT-Base')
        figure1, figure2 = report.figures
        figure1.show()
        figure2.show()
        print(report.latex)
    else:
        import robustnessgym as rg

        model_identifier = 'huggingface/textattack/bert-base-uncased-snli'
        task = rg.TernaryNaturalLanguageInference()
        if model_identifier.split("/")[0] == 'huggingface':
            model = rg.Model.huggingface(
                identifier="/".join(model_identifier.split("/")[1:]),
                task=task,
            )
        else:
            raise NotImplementedError

        # Create the test bench
        testbench = rg.TestBench(
            identifier='snli-nli-0.0.1dev',
            task=task,
            slices=[
                rg.Slice.from_dataset(identifier='snli-train',
                                      dataset=rg.Dataset.load_dataset('snli', split='train[:128]')).filter(
                    lambda example: example['label'] != -1),
                rg.Slice.from_dataset(identifier='snli-val',
                                      dataset=rg.Dataset.load_dataset('snli', split='validation[:128]')).filter(
                    lambda example: example['label'] != -1),
                rg.Slice.from_dataset(identifier='snli-test',
                                      dataset=rg.Dataset.load_dataset('snli', split='test[:128]')).filter(
                    lambda example: example['label'] != -1),
            ],
            dataset_id='snli'
        )

        # Create the report
        report = testbench.create_report(model=model,
                                         coerce_fn=functools.partial(rg.Model.remap_labels, label_map=[1, 2, 0]), )
        figure1, figure2 = report.figures
        if figure1:
            figure1.show()
        figure2.show()
    # st.write(figure1)
    # st.write(figure2)
