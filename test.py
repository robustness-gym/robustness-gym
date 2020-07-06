# from robustness_gym.dataset import Dataset
from robustness_gym.slicer import *
from robustness_gym.slice import *
import streamlit as st

st.title("Robustness Gym")

# dataset = nlp.load_dataset('boolq', split='train[:1%]')

# Load the boolq dataset
dataset = Dataset.load_dataset('boolq', split='train[:1%]')
dataset.initialize()

for e in tz.take(2, dataset):
    st.write(e)

st.write("Slice")
st.write(dataset[:2])

st.write(dataset)
