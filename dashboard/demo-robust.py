from collections import defaultdict

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
# st.sidebar.image("gym.png",width=100,format='PNG')
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
def generate_report(model, task, cols, data):

    score_colors = ['#EC7734', '#477CC9']
    score_color_complement = '#F3F4F7'
    text_fill_color = '#F3F4F7'
    text_line_color = '#BEC4CE'
    distribution_color_scale = [[0.0, '#FBF5F2'], [1.0, '#EC7734']]
    col_type_to_width = {
        'score': .6,
        'distribution': 0.35,
        'text': 0.25
    }
    col_names = [col['name'] for col in cols]
    n_cols = len(cols)
    col_widths = [col_type_to_width[col['type']] for col in cols]
    row_lengths = [len(list(row['data'].values())[0]) for row in data]
    group_names = [row['group'] for row in data]
    n_groups = len(data)
    chart_row_height = 24
    n_chart_rows = sum(row_lengths)
    group_space = 24
    header_space = 80
    height = n_chart_rows * chart_row_height + n_groups * group_space + header_space
    summary_data = defaultdict(dict)

    fig_detail = make_subplots(rows=n_groups,
                               row_titles=group_names,
                               cols=n_cols,
                               shared_yaxes=True,
                               subplot_titles=col_names,
                               horizontal_spacing=.035,
                               vertical_spacing=group_space / height,
                               row_width=list(reversed(row_lengths)),
                               column_width=col_widths
                               )

    hms = []
    coords = []
    for row_ndx, row in enumerate(data):
        group_name = row['group']
        for col_ndx, col in enumerate(cols):
            col_name = col['name']
            col_type = col['type']
            slices = [s + ' ' * 3 for s in row['slices']]
            x = row['data'][col_name]
            if col_type == 'score':
                max_score = col['max']
                fig_detail.add_trace(
                    go.Bar(
                        x=x,
                        y=slices,
                        orientation='h',
                        marker=dict(color=score_colors[col_ndx % 2]),
                        showlegend=False,
                        text=x,
                        textposition='inside',
                        width=.95,
                        textfont=dict(color='white')
                    ),
                    row=row_ndx+1, col=col_ndx+1
                )
                # Add marker for complementary gray fill
                fig_detail.add_trace(
                    go.Bar(
                        x=[max_score - x_i for x_i in x],
                        y=slices,
                        orientation='h',
                        marker=dict(color=score_color_complement),
                        showlegend=False,
                        width=.9
                    ),
                    row=row_ndx+1, col=col_ndx+1
                )
                # Accumulate summary statistics
                summary_data[col_name][group_name] = min(x)
            elif col_type == 'distribution':
                annotation_text = [[f"{int(round(z * 100)):d}" for z in rw] for rw in x]
                class_codes = col['class_codes']
                hm = ff.create_annotated_heatmap(x, x=class_codes, xgap=1, ygap=1,
                                                 annotation_text=annotation_text,
                                                 colorscale=distribution_color_scale,
                                                 zmin=0, zmax=1)
                hms.append(hm)
                # Save annotation data for special code related to heatmaps at end
                coords.append((n_cols * row_ndx) + col_ndx + 1)
                fig_detail.add_trace(
                    hm.data[0],
                    row=row_ndx + 1, col=col_ndx+1,
                )
            elif col_type == 'text':
                # Repurpose bar chart as text field.
                fig_detail.add_trace(
                    go.Bar(
                        x=[1] * len(x),
                        y=slices,
                        orientation='h',
                        marker=dict(color=text_fill_color,
                                    line=dict(width=0, color=text_line_color)),
                        showlegend=False,
                        text=x,
                        textposition='inside',
                        insidetextanchor='middle',
                        width=.9
                    ),
                    row=row_ndx + 1, col=col_ndx + 1
                )
            else:
                raise ValueError('Invalid col type')

    for row_ndx in range(n_groups):
        if row_ndx + 1 == n_groups:
            show_x_axis = True
        else:
            show_x_axis = False
        for col_ndx, col in enumerate(cols):
            fig_detail.update_yaxes(autorange='reversed')
            col_type = col['type']
            if col_type == 'score':
                min_score = col['min']
                max_score = col['max']
                fig_detail.update_xaxes(range=[min_score, max_score], row=row_ndx + 1, col=col_ndx + 1,
                                 tickvals=[min_score, max_score], showticklabels=show_x_axis)
            elif col_type == 'distribution':
                fig_detail.update_xaxes(row=row_ndx + 1, col=col_ndx + 1, showticklabels=show_x_axis)
            elif col_type == 'text':
                fig_detail.update_xaxes(range=[0, 1], row=row_ndx + 1, col=col_ndx + 1, showticklabels=False)
            else:
                raise ValueError('Invalid col type')


    fig_detail.update_layout(title=f'{task} / {model}',
                      height=height,
                      width=960,
                      barmode='stack',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      font=dict(size=13),
                      yaxis={'autorange':'reversed'}
                      )

    # Use low-level plotly interface to update padding / font size
    for a in fig_detail['layout']['annotations']:
        # If label for group
        if a['text'] in group_names:
            a['x'] = .99  # Add padding
        else:
            a['font'] = dict(size=14)  # Adjust font size for non-group labels

    # Due to a quirk of plotly, need to do some special low-level coding
    # Code based on https://community.plotly.com/t/how-to-create-annotated-heatmaps-in-subplots/36686/25
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
    score_names = []
    for col in cols:
        if col['type'] == 'score':
            score_names.append(col['name'])
    # TODO Support more than 2 scores
    n_cols = len(score_names)
    fig_summary = make_subplots(rows=1,
                                cols=n_cols,
                                subplot_titles=score_names,
                                specs=[[{'type': 'polar'}] * n_cols],
                                horizontal_spacing=.2,
                                column_width=[.35] * n_cols
                                )
    for i, score_name in enumerate(score_names):
        group_scores = [summary_data[score_name][group_name] for group_name in group_names]
        fig_summary.add_trace(go.Scatterpolar(
            name=score_name,
            r=group_scores,
            theta=group_names,
        ), 1, i+1)
    fig_summary.update_traces(fill='toself')

    st.write(fig_summary)
    st.write(fig_detail)


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
        # Mock data for testing
        #
        # text_cols defines format of each column in the report
        # 'type' is one of:
        #    'score': score from a particular model/metric
        #    'distribution': class distribution (heatmap)
        #    'text': free form text
        test_cols = [
            {'type': 'score', 'name': 'Accuracy', 'min': 0, 'max': 100},
            {'type': 'score', 'name': 'F1', 'min': 0, 'max': 100},
            {'type': 'distribution', 'name': 'Class %', 'class_codes': ['E', 'N', 'C']},
            {'type': 'distribution', 'name': 'Pred. class %', 'class_codes': ['E', 'N', 'C']},
            {'type': 'text', 'name': 'Size'},
        ]

        # Each item in test_data contains data for one group, e.g. the data for augmentations
        test_data = [
            # First group
            {'group': '',  # Name of group
             'slices': ['Test split'],  # Tests/slices for group
             'data': {  # Data, indexed by column name
                 'Accuracy': [55],
                 'F1': [45],
                 'Class %': [
                     [0.3, 0.4, 0.3]
                 ],
                 'Pred. class %': [
                     [0.27, 0.43, 0.3]
                 ],
                 'Size': '13K'
             }
             },
            # Second group, etc.
            {'group': 'Slices',
             'slices': ['Negation', 'Contains -ing', 'Temporal preposition', 'Ends with verb', 'slice 5',
                        'slice 6', 'slice 7'],
             'data': {
                 'Accuracy': [50, 40, 30, 20, 10, 30, 20],
                 'F1': [20, 30, 10, 75, 80, 75, 80],
                 'Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.2, 0.4, 0.4],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6]
                 ],
                 'Pred. class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.3, 0.4, 0.4],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                 ],
                 'Size': ['15K', '18K', '12K', '30K', '24K', '40K', '15K'],
             }
             },
            {'group': 'Augmentations',
             'slices': ['Augmentation 1', 'Augmentation 2', 'Augmentation 3',
                        'Augmentation 4', 'Augmentation 5', 'Augmentation 6', 'Augmentation 7'],
             'data': {
                 'Accuracy': [50, 40, 30, 20, 10, 20, 10],
                 'F1': [20, 30, 10, 75, 80, 75, 80],
                 'Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.1, 0.6, 0.3],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6]
                 ],
                 'Pred. class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.2, 0.5, 0.3],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6]
                 ],
                 'Size': ['230', '521', '1K', '100', '2K', '3K', '210'],
             }
             },
            {'group': 'TextAttack',
             'slices': ['Textfooler', 'Hotflip', 'Morpheus', 'Seq2Sick', 'Hotflip 2',
                        'Morpheus 2', 'Seq2Sick 2'],
             'data': {
                 'Accuracy': [50, 40, 30, 20, 10, 20, 10],
                 'F1': [20, 30, 10, 75, 80, 10, 75, 80],
                 'Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.8, 0.1, 0.1],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                 ],
                 'Pred. class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.3, 0.4, 0.2],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.1, 0.8],
                     [0.1, 0.3, 0.6],
                 ],
                 'Size': ['3.1K', '15K', '412', '8.1K', '15K', '412', '8.1K'],
             }
             },
            {'group': 'Eval Sets',
             'slices': ['Eval set 1', 'Eval set 2', 'Eval set 3', 'Eval set 4', 'Eval set 5',
                        'Eval set 6', 'Eval set 7'],
             'data': {
                 'Accuracy': [50, 40, 30, 20, 10, 20, 10],
                 'F1': [20, 30, 10, 75, 80, 10, 75, 80],
                 'Class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.2, 0.6, 0.2],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                 ],
                 'Pred. class %': [
                     [0.1, 0.3, 0.6],
                     [0.3, 0.5, 0.2],
                     [0.4, 0.4, 0.2],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                     [0.1, 0.3, 0.6],
                 ],
                 'Size': ['3.1K', '15K', '412', '8.1K', '15K', '412', '8.1K'],
             }
             }
        ]
        st.header("Model Robustness report")
        task = st.sidebar.selectbox("Choose dataset", ["SNLI", "SST-2", "Summarization","MRPC"])  #call Dataset.list_datasets
        model = st.sidebar.selectbox("Choose model", ["BERT-Base", "RoBERTa-Base", "BART", "T5"])
        generate_report(model, task, test_cols, test_data)


