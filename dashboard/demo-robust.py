import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import plotly.express as px
import streamlit as st
import SessionState

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
    # TODO This fn will call Augmentation.report, TextAttack.report and sfs report that will get the preds for the task
    # call get_robust_slices(task) will return task_slices
    task_slices = ['gender words', 'length <10', 'negation', 'ends in a verb', 'temporal preposition', 'has k colors']
    # call get_acc_slices(task_slices) will return acc_slices
    # THe above two can be consolidated into one fn that returns a pd.DataFrame that includes slices, perf, metric, ...
    acc_slices = [82.5, 68.9, 52.7, 73.7, 79.5, 83.6]    #placeholder
    generate_slice_chart(acc_slices, task_slices)


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
