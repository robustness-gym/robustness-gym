import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
    classes = ['E', 'N', 'C']

    metrics = ['accuracy', 'f1']
    metric_colors = ['#EC7734', '#477CC9']

    data_original = {'group': 'Original data',
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
                  'slices': ['Negation   ', 'Contains -ing   ', 'Temporal preposition   ', 'Ends with verb   '
                      , 'slice 5   ', 'slice 6   ', 'slice 7   '],
                  'class_dist': [
                      [0.1, 0.3, 0.6],
                      [0.3, 0.5, 0.2],
                      [0.1, 0.6, 0.3],
                      [0.1, 0.3, 0.6],
                      [0.1, 0.3, 0.6],
                      [0.1, 0.3, 0.6],
                      [0.1, 0.3, 0.6],
                  ],
                  'pred_dist': [
                      [0.1, 0.3, 0.6],
                      [0.3, 0.5, 0.2],
                      [0.2, 0.5, 0.3],
                      [0.1, 0.3, 0.6],
                      [0.1, 0.3, 0.6],
                      [0.1, 0.3, 0.6],
                      [0.1, 0.3, 0.6],
                  ],
                  'accuracy': [50, 40, 30, 20, 10, 20, 10],
                  'f1': [20, 30, 10, 75, 80, 75, 80],
                  'size': ['230', '521', '1K', '100', '2K', '3K', '210'],
                  'cost': ['low', 'low', 'low', 'low', 'med', 'med', 'med']}

    data_augmentation = {'group': 'Augmentations',
                         'slices': ['Augmentation 1   ', 'Augmentation 2   ', 'Augmentation 3   ',
                                    'Augmentation 4   ', 'Augmentation 5   ', 'Augmentation 6   ', 'Augmentation 7   '],
                         'class_dist': [
                             [0.1, 0.3, 0.6],
                             [0.3, 0.5, 0.2],
                             [0.2, 0.4, 0.4],
                             [0.1, 0.3, 0.6],
                             [0.1, 0.3, 0.6],
                             [0.1, 0.3, 0.6],
                             [0.1, 0.3, 0.6],
                         ],
                         'pred_dist': [
                             [0.1, 0.3, 0.6],
                             [0.3, 0.5, 0.2],
                             [0.3, 0.4, 0.4],
                             [0.1, 0.3, 0.6],
                             [0.1, 0.3, 0.6],
                             [0.1, 0.3, 0.6],
                             [0.1, 0.3, 0.6],
                         ],
                         'accuracy': [50, 40, 30, 20, 10, 30, 20],
                         'f1': [20, 30, 10, 75, 80, 75, 80],
                         'size': ['15K', '18K', '12K', '30K', '24K', '40K', '15K'],
                         'cost': ['low', 'low', 'low', 'low', 'low', 'low', 'low']}

    data_eval_set = {'group': 'Eval sets',
                     'slices': ['Eval set 1   ', 'Eval set 2   ', 'Eval set 3   ', 'Eval set 4   ', 'Eval set 5   ',
                                'Eval set 6   ', 'Eval set 7   '],
                     'class_dist': [
                         [0.1, 0.3, 0.6],
                         [0.3, 0.5, 0.2],
                         [0.2, 0.6, 0.2],
                         [0.1, 0.3, 0.6],
                         [0.1, 0.3, 0.6],
                         [0.1, 0.3, 0.6],
                         [0.1, 0.3, 0.6],
                     ],
                     'pred_dist': [
                         [0.1, 0.3, 0.6],
                         [0.3, 0.5, 0.2],
                         [0.4, 0.4, 0.2],
                         [0.1, 0.3, 0.6],
                         [0.1, 0.3, 0.6],
                         [0.1, 0.3, 0.6],
                         [0.1, 0.3, 0.6],
                     ],
                     'accuracy': [50, 40, 30, 20, 10, 20, 10],
                     'f1': [20, 30, 10, 75, 80, 75, 80],
                     'size': ['3.1K', '15K', '412', '8.1K', '15K', '412', '8.1K'],
                     'cost': ['low', 'low', 'low', 'low', 'low', 'low', 'low']}

    data_adversarial = {'group': 'TextAttack',
                        'slices': ['Textfooler   ', 'Hotflip   ', 'Morpheus   ', 'Seq2Sick   ', 'Hotflip 2  ',
                                   'Morpheus 2   ', 'Seq2Sick 2  '],
                        'class_dist': [
                            [0.1, 0.3, 0.6],
                            [0.3, 0.5, 0.2],
                            [0.8, 0.1, 0.1],
                            [0.1, 0.3, 0.6],
                            [0.1, 0.3, 0.6],
                            [0.1, 0.3, 0.6],
                            [0.1, 0.3, 0.6],
                        ],
                        'pred_dist': [
                            [0.1, 0.3, 0.6],
                            [0.3, 0.5, 0.2],
                            [0.3, 0.4, 0.2],
                            [0.1, 0.3, 0.6],
                            [0.1, 0.3, 0.6],
                            [0.1, 0.1, 0.8],
                            [0.1, 0.3, 0.6],
                        ],
                        'accuracy': [50, 40, 30, 20, 10, 20, 10],
                        'f1': [20, 30, 10, 75, 80, 10, 75, 80],
                        'size': ['3.1K', '15K', '412', '8.1K', '15K', '412', '8.1K'],
                        'cost': ['high', 'medium', 'low', 'high', 'medium', 'low', 'high']}

    data = [data_original, data_slice, data_augmentation, data_adversarial, data_eval_set]
    groups = [d['group'] for d in data]

    n_metrics = len(metrics)
    n_cols = n_metrics + 3
    n_groups = len(groups)

    subplot_titles = ([metric.capitalize() for metric in metrics] + \
                      ['Class %',
                       'Pred. Class %',
                       'Size'])
    row_width = [7] * (n_groups - 1) + [1]
    fig = make_subplots(rows=n_groups,
                        row_titles=[''] + groups[1:],
                        cols=n_cols,
                        shared_yaxes=True,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=.035,
                        vertical_spacing=.03,
                        row_width=row_width,
                        column_width=[.6, .6, .35, .35, .25]
                        )

    dist_legend_shown = False
    row = 1
    annots = []
    hms = []
    coords = []
    for d in data:
        col = 1
        for metric, color in zip(metrics, metric_colors):
            fig.add_trace(
                go.Bar(
                    x=d[metric],
                    y=d['slices'],
                    orientation='h',
                    marker=dict(color=color),
                    showlegend=False,
                    text=d[metric],
                    textposition='inside',
                    width=.9,
                    textfont=dict(color='white')
                ),
                row=row, col=col
            )
            # Add marker for complementary gray fill
            fig.add_trace(
                go.Bar(
                    x=[100 - val for val in d[metric]],
                    y=d['slices'],
                    orientation='h',
                    marker=dict(color='#F3F4F7'),
                    showlegend=False,
                    width=.9
                ),
                row=row, col=col
            )
            col += 1

        for dist_type in ['class_dist', 'pred_dist']:
            annotation_text = [[f"{int(round(x * 100)):d}" for x in rw] for rw in d[dist_type]]
            hm = ff.create_annotated_heatmap(d[dist_type], x=classes, xgap=1, ygap=1,
                                             annotation_text=annotation_text,
                                             colorscale=[[0.0, '#FBF5F2'], [1.0, '#EC7734']],
                                             zmin=0, zmax=1
                                             )
            hms.append(hm)
            coords.append(n_cols * (row - 1) + col)
            fig.add_trace(
                hm.data[0],
                row=row, col=col,
            )
            col += 1

        # Repurpose bar chart as text field.
        fig.add_trace(
            go.Bar(
                x=[1] * len(d['size']),
                y=d['slices'],
                orientation='h',
                marker=dict(color='#F3F4F7',
                            line=dict(width=0, color='#BEC4CE')),
                showlegend=False,
                text=d['size'],
                textposition='inside',
                insidetextanchor='middle',
                width=.95
            ),
            row=row, col=col
        )
        col += 1
        row += 1

    for row in range(1, n_groups + 1):
        if row == n_groups:
            show_x_axis = True
        else:
            show_x_axis = False
        fig.update_xaxes(range=[0, 100], row=row, col=1, tickvals=[0, 100], showticklabels=show_x_axis)
        fig.update_xaxes(range=[0, 100], row=row, col=2, tickvals=[0, 100], showticklabels=show_x_axis)
        fig.update_xaxes(row=row, col=3, showticklabels=show_x_axis)
        fig.update_xaxes(row=row, col=4, showticklabels=show_x_axis)
        fig.update_xaxes(range=[0, 1], row=row, col=5, showticklabels=False)
        fig.update_xaxes(range=[0, 1], row=row, col=6, showticklabels=False)

    fig.update_layout(title=f'{task} / {model}',
                      height=900,
                      width=960,
                      barmode='stack',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      font=dict(size=13),
                      )

    # Use low-level plotly interface to update padding / font size
    for a in fig['layout']['annotations']:
        # If label for group
        if a['text'] in groups:
            a['x'] = .99  # Add padding
        else:
            a['font'] = dict(size=14)  # Adjust font size for non-group labels

    # Due to a quirk of plotly, need to do some special low-level coding
    # Code based on https://community.plotly.com/t/how-to-create-annotated-heatmaps-in-subplots/36686/25
    newfont = [go.layout.Annotation(font_size=14)] * len(fig.layout.annotations)
    fig_annots = [newfont] + [hm.layout.annotations for hm in hms]
    for j in range(1, len(fig_annots)):
        for k in range(len(fig_annots[j])):
            coord = coords[j - 1]
            fig_annots[j][k]['xref'] = f'x{coord}'
            fig_annots[j][k]['yref'] = f'y{coord}'
            fig_annots[j][k]['font_size'] = 11

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
