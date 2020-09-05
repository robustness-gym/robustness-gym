import datetime
import functools
from collections import defaultdict
from pathlib import Path
from random import randint

import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
# import SessionState
import sys

sys.path.append('..')
from robustness_gym import *

st.title("Robustness gym")
# st.sidebar.image("gym.png",width=100,format='PNG')
view = st.sidebar.radio("Select view", ("Slice view", "Model view"))


# html_view_tab = """<div class="tab"><button id="sliceview" onclick="document.getElementById("sliceview")">Slice View</button><button class="tabview" onclick="openView(event, "Model View")">Model View</button></div>
# """
# st.markdown(html_view_tab, unsafe_allow_html=True)
# st.markdown(js_view_tab, unsafe_allow_html=True)
# this is for slice size slider
# widget = st.empty()
# if st.button('Increment position'):
#    state.position += 1


def generate_slice_chart(acc_slices, task_slices):
    df = pd.DataFrame(dict(
        acc=acc_slices,
        task_slices=task_slices))
    fig = px.line_polar(df, r='acc', theta='task_slices', line_close=True, hover_data=['acc'])
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True
            )),
        showlegend=False
    )
    st.write(fig)


if view == "Slice view":
    st.header("Slice view")
    task = st.sidebar.selectbox("Choose dataset",
                                ["SNLI", "SST-2", "Summarization", "MRPC"])  # call Dataset.list_datasets
    slider_pos = st.sidebar.slider('Min # examples per slice', 10, 100, 50)
    # call Slice.nlpviewer(slider_pos)
else:

    model_view_type = st.sidebar.selectbox("Choose report type", ["Standard", "Custom"])

    st.header("Model Robustness report")
    # task = st.sidebar.selectbox("Choose dataset", ["SNLI", "SST-2", "Summarization","MRPC"])  #call Dataset.list_datasets

    # Select a task
    task_identifier = st.sidebar.selectbox("Choose a task", ["TernaryNaturalLanguageInference"])
    task = Task.create(task=task_identifier)

    # Select a dataset
    dataset = st.sidebar.selectbox("Choose dataset", list(task.datasets()))

    # Select a model
    model_identifier = st.sidebar.selectbox("Choose model", [
        'huggingface/textattack/bert-base-uncased-snli'])  # ["BERT-Base", "RoBERTa-Base", "BART", "T5"])

    if model_view_type == "Custom":
        if model_identifier.split("/")[0] == 'huggingface':
            model = Model.huggingface(
                identifier="/".join(model_identifier.split("/")[1:]),
                task=task,
            )
        else:
            raise NotImplementedError

        # Create the test bench
        testbench = TestBench(
            identifier='snli-nli-0.0.1dev',
            task=task,
            slices=[
                Slice.from_dataset(identifier='snli-train',
                                   dataset=Dataset.load_dataset('snli', split='train[:128]')).filter(
                    lambda example: example['label'] != -1),
                Slice.from_dataset(identifier='snli-val',
                                   dataset=Dataset.load_dataset('snli', split='validation[:128]')).filter(
                    lambda example: example['label'] != -1),
                Slice.from_dataset(identifier='snli-test',
                                   dataset=Dataset.load_dataset('snli', split='test[:128]')).filter(
                    lambda example: example['label'] != -1),
            ],
            dataset_id='snli'
        )

        # TODO Use actual available slices
        group_slices = [
            ('Slices', [{'name': f'Slice_{i}', 'size': randint(10, 100)} for i in range(1, 21)]),
            ('Augmentations', [{'name': f'Augmentation_{i}', 'size': randint(10, 100)} for i in range(1, 21)]),
            ('Adversarial', [{'name': f'Adversarial_{i}', 'size': randint(10, 100)} for i in range(1, 21)]),
            ('Eval Sets', [{'name': f'EvalSet_{i}', 'size': randint(10, 100)} for i in range(1, 21)])
        ]
        group_to_selected = {}
        for group, slices in group_slices:
            # filtered_slices = [slice['name'] for slice in slices if slice['size'] >= slider_pos]
            # group_to_selected[group] = st.multiselect(group, filtered_slices)
            group_to_selected[group] = st.sidebar.multiselect(group, [slice['name'] for slice in slices])
            print('group')
            print(group_to_selected[group])

        st.write("**Display columns**")

        metrick_checklist = {}
        for metric in task.metrics:
            metrick_checklist[metric] = st.checkbox(metric.replace('_', ' ').title(), value=metric)
        checked_metrics = [m for m, checked in metrick_checklist.items() if checked]
        st.text("")

        if st.button('Generate report'):
            report = testbench.create_report(model=model,
                                             coerce_fn=functools.partial(Model.remap_labels, label_map=[1, 2, 0]),
                                             metric_ids=checked_metrics)
            fig1, fig2 = report.figures()
            if fig1 is not None:
                fig1
            fig2
    else:

        # Load the model
        # TODO(karan): generalize this
        if model_identifier.split("/")[0] == 'huggingface':
            model = Model.huggingface(
                identifier="/".join(model_identifier.split("/")[1:]),
                task=task,
            )
        else:
            raise NotImplementedError

        # Create the test bench
        testbench = TestBench(
            identifier='snli-nli-0.0.1dev',
            task=task,
            slices=[
                Slice.from_dataset(identifier='snli-train',
                                   dataset=Dataset.load_dataset('snli', split='train[:128]')).filter(
                    lambda example: example['label'] != -1),
                Slice.from_dataset(identifier='snli-val',
                                   dataset=Dataset.load_dataset('snli', split='validation[:128]')).filter(
                    lambda example: example['label'] != -1),
                Slice.from_dataset(identifier='snli-test',
                                   dataset=Dataset.load_dataset('snli', split='test[:128]')).filter(
                    lambda example: example['label'] != -1),
            ],
            dataset_id='snli'
        )

        # Create the report
        report = testbench.create_report(model=model,
                                         coerce_fn=functools.partial(Model.remap_labels, label_map=[1, 2, 0]),)
        #
        # text_cols defines format of each column in the report
        # 'type' is one of:
        #    'score': score from a particular model/metric
        #    'distribution': class distribution (heatmap)
        #    'text': free form text
        # TODO(karan): this will come from the report
        # test_cols = [
        #     {'type': 'score', 'name': 'accuracy', 'min': 0, 'max': 1},
        #     {'type': 'score', 'name': 'f1', 'min': 0, 'max': 1},
        #     # {'type': 'distribution', 'name': 'Class %', 'class_codes': ['E', 'N', 'C']},
        #     # {'type': 'distribution', 'name': 'Pred. class %', 'class_codes': ['E', 'N', 'C']},
        #     {'type': 'text', 'name': 'Size'},
        # ]
        #
        # # TODO(karan): remove hax
        # test_data = report[-1:] * 4  # + report[1]
        #
        # generate_report(model, task, dataset, test_cols, test_data)
        fig1, fig2 = report.figures()
        if fig1 is not None:
            fig1
        fig2
        if st.button('Generate appendix'):
            tstamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            reports_path = Path(__file__).parent.parent / 'reports'
            output_path = reports_path / f'appendix_{tstamp}'
            report.write_appendix(output_path)
            f'Files written to {output_path}. See README.txt in appropriate subdirectory.'
