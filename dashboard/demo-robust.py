import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# import SessionState

st.title("Robustness gym")
st.sidebar.image("gym.png",width=100,format='PNG')
view = st.sidebar.radio("Select view", ("Slice view","Model view"))
#html_view_tab = """<div class="tab"><button id="sliceview" onclick="document.getElementById("sliceview")">Slice View</button><button class="tabview" onclick="openView(event, "Model View")">Model View</button></div>
#"""
#st.markdown(html_view_tab, unsafe_allow_html=True)
#st.markdown(js_view_tab, unsafe_allow_html=True)
#this is for slice size slider
#widget = st.empty()
#if st.button('Increment position'):
#    state.position += 1


def generate_slice_chart(acc_slices, task_slices):
    df = pd.DataFrame(dict(
        acc= acc_slices,
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

@st.cache(suppress_st_warning=True)
def generate_report(model, task):
    f'{task} / {model}'
    classes = ['Entail', 'Neutral', 'Contradict']

    metrics = ['accuracy', 'f1']

    data_original = {'group': 'Original',
                     'slices': ['Test split   '],
                     'class_dist': [
                         [0.3, 0.4, 0.3],
                     ],
                     'pred_dist': [
                         [0.27, 0.43, 0.3],
                     ],
                     'accuracy': [55],
                     'f1': [45],
                     'size': ['13K'],
                     'cost': ['']}

    data_slice = {'group': 'Slices',
                  'slices': ['Negation   ', 'Contains -ing   ', 'Temporal preposition   ', 'Ends with verb   '],
                  'class_dist': [
                      [0.1, 0.3, 0.6],
                      [0.3, 0.5, 0.2],
                      [0.8, 0.1, 0.1],
                      [0.1, 0.3, 0.6],
                  ],
                  'pred_dist': [
                      [0.1, 0.3, 0.6],
                      [0.3, 0.5, 0.2],
                      [0.8, 0.1, 0.1],
                      [0.1, 0.3, 0.6],
                  ],
                  'accuracy': [50, 40, 30, 20, 10],
                  'f1': [20, 30, 10, 75, 80],
                  'size': ['20%', '5%', '13%', '14%'],
                  'cost': ['low', 'low', 'low', 'low']}

    data_augmentation = {'group': 'Augmentations',
                         'slices': ['Augmentation 1   ', 'Augmentation 2   ', 'Augmentation 3   ',
                                    'Augmentation 4   '],
                         'class_dist': [
                             [0.1, 0.3, 0.6],
                             [0.3, 0.5, 0.2],
                             [0.8, 0.1, 0.1],
                             [0.1, 0.3, 0.6],
                         ],
                         'pred_dist': [
                             [0.1, 0.3, 0.6],
                             [0.3, 0.5, 0.2],
                             [0.8, 0.1, 0.1],
                             [0.1, 0.3, 0.6],
                         ],
                         'accuracy': [50, 40, 30, 20, 10],
                         'f1': [20, 30, 10, 75, 80],
                         'size': ['1.4x', '2x', '4x', '3x'],
                         'cost': ['low', 'low', 'low', 'low']}

    data_eval_set = {'group': 'Eval sets',
                     'slices': ['Eval set 1   ', 'Eval set 2   ', 'Eval set 3   ', 'Eval set 4   '],
                     'class_dist': [
                         [0.1, 0.3, 0.6],
                         [0.3, 0.5, 0.2],
                         [0.8, 0.1, 0.1],
                         [0.1, 0.3, 0.6],
                     ],
                     'pred_dist': [
                         [0.1, 0.3, 0.6],
                         [0.3, 0.5, 0.2],
                         [0.8, 0.1, 0.1],
                         [0.1, 0.3, 0.6],
                     ],
                     'accuracy': [50, 40, 30, 20, 10],
                     'f1': [20, 30, 10, 75, 80],
                     'size': ['3.1K', '15K', '412', '8.1K'],
                     'cost': ['low', 'low', 'low', 'low']}

    data_adversarial = {'group': 'TextAttack',
                        'slices': ['Textfooler   ', 'Hotflip   ', 'Morpheus   ', 'Seq2Sick   '],
                        'class_dist': [
                            [0.1, 0.3, 0.6],
                            [0.3, 0.5, 0.2],
                            [0.8, 0.1, 0.1],
                            [0.1, 0.3, 0.6],
                        ],
                        'pred_dist': [
                            [0.1, 0.3, 0.6],
                            [0.3, 0.5, 0.2],
                            [0.8, 0.1, 0.1],
                            [0.1, 0.3, 0.6],
                        ],
                        'accuracy': [50, 40, 30, 20, 10],
                        'f1': [20, 30, 10, 75, 80],
                        'size': ['3.1K', '15K', '412', '8.1K'],
                        'cost': ['high', 'medium', 'low', 'high']}

    data = [data_original, data_slice, data_augmentation, data_eval_set, data_adversarial]
    groups = [d['group'] for d in data]

    min_acc = [.7, .6, .8, .9, .4]
    min_f1 = [.6, .9, .4, .9, .5]


    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=min_acc,
        theta=groups,
        fill='toself',
        name='Accuracy'
    ))
    fig.add_trace(go.Scatterpolar(
        r=min_f1,
        theta=groups,
        fill='toself',
        name='F1'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        height=700,
        showlegend=True,
        title='Minimum performance by group'
    )

    st.write(fig)

    dist_colors = ['#FDF0E7', '#F8C4A0', '#E8823B']

    subplot_titles = sum([[metric.capitalize(), ''] for metric in metrics], []) + \
                     ['Class Dist.', None,
                      'Prediction Dist.', None,
                      'Size',
                      'Cost']

    n_metrics = len(metrics)
    specs = len(groups) * [[{'colspan': 2}, {}] * n_metrics + [{'colspan': 2}, {}, {'colspan': 2}, {}, {}, {}, {'r': .07}]]

    fig = make_subplots(rows=len(groups),
                        row_titles=groups,
                        cols=2 * n_metrics + 7,
                        shared_yaxes=True,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=.04,
                        vertical_spacing=.04,
                        specs=specs,
                        row_width=[4, 4, 4, 4, 1],
                        )

    print(fig.layout)

    dist_legend_shown = False
    row = 1
    for d in data:
        col = 1
        for metric in metrics:
            # Add marker for metric
            fig.add_trace(
                go.Bar(
                    x=d[metric],
                    y=d['slices'],
                    orientation='h',
                    marker=dict(color='#477CC9',
                                line=dict(width=1, color='#477CC9')),
                    showlegend=False,
                    text=d[metric],
                    textposition='outside',
                    width=.7
                ),
                row=row, col=col
            )
            # Add marker for complementary gray fill
            fig.add_trace(
                go.Bar(
                    x=[100 - val for val in d[metric]],
                    y=d['slices'],
                    orientation='h',
                    marker=dict(color='#F3F4F7',
                                line=dict(width=1, color='#BEC4CE')),
                    showlegend=False,
                    width=.7
                ),
                row=row, col=col
            )
            col += 2

        # Add class distributions
        for dist_type in ['class_dist', 'pred_dist']:
            for i, class_ in enumerate(classes):
                dist = [class_dist[i] for class_dist in d[dist_type]]
                fig.add_trace(
                    go.Bar(
                        x=dist,
                        y=d['slices'],
                        orientation='h',
                        marker=dict(color=dist_colors[i],
                                    line=dict(width=1, color=dist_colors[-1])),
                        showlegend=not dist_legend_shown,
                        width=.7,
                        # legendgroup=1,
                        name=class_
                    ),
                    row=row, col=col
                )
            dist_legend_shown = True
            col += 2

        # Add size. Repurpose bar chart as text field.
        fig.add_trace(
            go.Bar(
                x=[1] * 4,
                y=d['slices'],
                orientation='h',
                marker=dict(color='#F3F4F7',
                            line=dict(width=0, color='#BEC4CE')),
                showlegend=False,
                text=d['size'],
                textposition='inside',
            ),
            row=row, col=col
        )
        col += 1

        # Add cost. Repurpose bar chart as text field.
        fig.add_trace(
            go.Bar(
                x=[1] * 4,
                y=d['slices'],
                orientation='h',
                marker=dict(color='#F3F4F7',
                            line=dict(width=0, color='#BEC4CE')),
                showlegend=False,
                text=d['cost'],
                textposition='inside',
                insidetextanchor='middle'
            ),
            row=row, col=col
        )
        row += 1

    for row in range(1, len(groups) + 1):
        if row == len(groups):
            show_x_axis = True
        else:
            show_x_axis = False
        fig.update_xaxes(range=[0, 100], row=row, col=1, tickvals=[0, 100], showticklabels=show_x_axis)
        fig.update_xaxes(range=[0, 100], row=row, col=3, tickvals=[0, 100], showticklabels=show_x_axis)
        fig.update_xaxes(range=[0, 1], row=row, col=5, tickvals=[0, 1], showticklabels=show_x_axis)
        fig.update_xaxes(range=[0, 1], row=row, col=7, tickvals=[0, 1], showticklabels=show_x_axis)
        fig.update_xaxes(range=[0, 1], row=row, col=9, showticklabels=False)
        fig.update_xaxes(range=[0, 1], row=row, col=10, showticklabels=False)
    fig.update_layout(title='Details',
                      height=750,
                      width=1000,
                      barmode='stack',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      )
    st.write(fig)


if view == "Slice view":
    st.header("Slice view")
    task = st.sidebar.selectbox("Choose dataset",
                                ["SNLI", "SST-2", "Summarization", "MRPC"])  # call Dataset.list_datasets
    slider_pos = st.sidebar.slider('Min # examples per slice', 10, 100, 50)
    # call Slice.nlpviewer(slider_pos)
else:
    model_view_type = st.sidebar.selectbox("Choose model view type", ["Robustness report", "Model analysis"])
    if model_view_type == "Model analysis":
        st.header("Model Analysis")
        # st.markdown(html_view_tab,unsafe_allow_html=True)
        task = st.sidebar.selectbox("Choose dataset",
                                    ["SNLI", "SST-2", "Summarization", "MRPC"])  # call Dataset.list_datasets
        model = st.sidebar.selectbox("Choose model", ["BERT-Base", "RoBERTa-Base", "BART", "T5"])
        slider_pos = st.sidebar.slider('Min # examples per slice', 10, 100, 50)
        # get_all_slices(task) returns df_slices
        df_sfs = pd.DataFrame({
            'sfs': ['negation','length < 10','temporal preposition', 'ends with verb', 'gendered words'],
            'size': [500, 79, 80, 56, 150],
            'perf': [75, 83, 57, 49, 66]

        })
        # 'adv': {'Weight poisoning', 'Adversarial triggers', 'Hotflip', 'Textfool'},
        # 'aug': {'Counterfactual', 'Backtranslation', 'Substitution', 'EDA'},
        # 'eval': {'Stress set', 'Contrast set', 'HANS', 'ANLI'}
        filtered_slices = df_sfs.loc[df_sfs['size'] >= slider_pos]
        selected_slices = st.multiselect('Data slices', filtered_slices['sfs'])
        selected_slices_frame = df_sfs.loc[selected_slices]
        #st.write(selected_slices_frame.head())
        fig = px.line_polar(filtered_slices, r='perf', theta='sfs', line_close=True)
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
        fig.show()
        st.write(fig)
        # dynamic_plot

    else:
        st.header("Model Robustness report")
        task = st.sidebar.selectbox("Choose dataset", ["SNLI", "SST-2", "Summarization","MRPC"])  #call Dataset.list_datasets
        model = st.sidebar.selectbox("Choose model", ["BERT-Base", "RoBERTa-Base", "BART", "T5"])
        generate_report(model, task)
