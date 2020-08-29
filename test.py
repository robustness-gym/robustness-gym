# from robustness_gym.dataset import Dataset
from robustness_gym.slicemaker import *
from robustness_gym.slice import *
from robustness_gym.curators.filters.phrase import *
from robustness_gym.slicers.augmentations.eda import *
import streamlit as st
import pandas as pd

st.title("Robustness Gym")


@st.cache(allow_output_mutation=True)
def load_data():
    # Normally this is how a dataset is loaded
    # dataset = nlp.load_dataset('boolq', split='train[:1%]')

    # Load the boolq dataset
    dataset = Dataset.load_dataset('boolq', split='train[:1%]')
    dataset.initialize()

    return dataset


dataset = load_data()

for e in tz.partition_all(2, dataset):
    print(len(e))
    # e = list(map(lambda d: tz.keyfilter(lambda k: k in ['question', 'answer', ], d), e))
    print(e)
    print("")
    print(tz.merge_with(tz.identity, *e))

# st.write("Slice")
# st.write(dataset[:2])
# st.write(dataset)

# Slicers


# Slice a batch
has_phrase_slicer = HasPhrase(phrases=['iran', 'afghanistan', 'do', 'samaritan'])
batch, slices, slice_labels = has_phrase_slicer(batch=dataset[:2], keys=['question'])
st.write(pd.DataFrame(slice_labels, columns=has_phrase_slicer.headers))

has_all_phrases_slicer = HasAllPhrases(phrases=['iran', 'afghanistan', 'samaritan'])

# Need this
# -------------------------
# from ops import morphological
# morphological_slicer = HasAnyPhrase(phrases=morphological("do")) # ['do', 'does', 'done']
# dataset.slice(morphological_slicer)
# SlicerOp: SlicerUnion, SlicerIntersection, SlicerSymmetricDiff, ...
# HasAnyPhrase = SlicerUnion([HasPhrase(phrases=['do']),
#                             HasPhrase(phrases=['does']),
#                             HasPhrase(phrases=['done'])])

# Good features to have
# 'do' -> 3 times

# Need this. Done
# --------------------
# dataset.save('my_boolq')
# Dataset.from_disk('my_boolq')

# Need
# - Documentation

# Jupyter notebook
# ---------------------
# - Concrete examples
#   Deep dive BERT, NLI
# -


# You can do this
# dataset = Dataset.from_batches(...)
# dataset.slice([HasPhrase(phrases=['iran']), HasPhrase(phrases=['afghan'])])

# augmentation = AugmentationOfSomeKind(...)
# slices, slice_labels = augmentation(batch=dataset[:2], keys=['question'])
# slices, slice_labels = augmentation(dataset=dataset, keys=['question'])

st.write(dataset.info)
st.write(dataset.split)

for header, slice in zip(has_phrase_slicer.headers, slices):
    st.write(header)
    st.write(slice)

# Slice a dataset
dataset, slices, slice_labels = has_phrase_slicer.slice_dataset(dataset, keys=['question'])
st.write("Updated Dataset?")
for e in tz.take(20, dataset):
    st.write(e['slices'])

# Save updated dataset and load it back
dataset.save('/Users/krandiash/Desktop/my_boolq')
old_dataset = Dataset.load('/Users/krandiash/Desktop/my_boolq')
st.write(dataset.split)
st.write(old_dataset.split)
for e in tz.take(20, old_dataset):
    st.write(e['slices'])

for header, slice in zip(has_phrase_slicer.headers, slices):
    st.write(header)
    for example in slice:
        st.write(example['question'])
st.write(pd.DataFrame(slice_labels, columns=has_phrase_slicer.headers))

has_any_phrase_slicer = HasAnyPhrase(phrases=['iran', 'afghanistan'])
batch, _, slice_labels = has_any_phrase_slicer.process_batch(batch=dataset[:2], keys=['question'])
st.write(pd.DataFrame(slice_labels, columns=has_any_phrase_slicer.headers))

has_all_phrases_slicer = HasAllPhrases(phrases=['iran', 'afghanistan', 'samaritan'])
batch, _, slice_labels = has_all_phrases_slicer.process_batch(batch=dataset[:2], keys=['question'])
st.write(pd.DataFrame(slice_labels, columns=has_all_phrases_slicer.headers))

# Augmentation
st.write("Easy Data Augmentation")
eda = EasyDataAugmentation(num_aug=3)
dataset, slices, slice_labels = eda(dataset, keys=['question'])
st.write(dataset)
st.write(slices)
st.write(slices[0])
for e in tz.take(1, slices[0]):
    st.write(e)

for e in tz.take(1, slices[1]):
    st.write(e)

for e in tz.take(2, dataset):
    st.write(e)

# Combine slices
st.write(Slice.interleave(slices))
st.write(Slice.chain(slices))

st.write(slice_labels)

# Simple way to create a generic slicer from a fn
example = SliceMaker(slice_batch_fn=lambda b, k: st.write("My slicer just ran."))
example.process_batch(None, None)

union_has_phrase = FilterMixin.union(*[
    HasPhrase(['iran', 'samaritan']),
    HasPhrase(['do']),
])

batch, _, slice_labels = union_has_phrase(dataset[:2], keys=['question'])

st.write(slice_labels)
st.write(batch)

complicated_has_phrase = FilterMixin.union(
    FilterMixin.intersection(
        HasPhrase(['iran']),
        HasPhrase(['afghanistan']),
    ),
    HasPhrase(['samaritan']),
    # name='MySlicer',
)

batch, _, slice_labels = complicated_has_phrase(dataset[:2], keys=['question'])

st.write(slice_labels)
st.write(batch)

# Slice a batch
has_phrase_slicer_1 = HasPhrase(phrases=['iran', 'afghanistan'])
batch, slices, slice_labels = has_phrase_slicer_1(batch=dataset[:2], keys=['question'])
st.write(pd.DataFrame(slice_labels, columns=has_phrase_slicer.headers))
# HasPhrase-1: [0, 1]
# {'iran': 0, 'afghanistan': 1}

# HasPhrase: [0, 1]

# {'HasPhrase': {'iran': 0, 'afghanistan': 1}}

has_phrase_slicer_2 = HasPhrase(phrases=['do good', 'samaritan'])
batch, slices, slice_labels = has_phrase_slicer_2(batch=dataset[:2], keys=['question'])
st.write(pd.DataFrame(slice_labels, columns=has_phrase_slicer.headers))
# HasPhrase-1: [0, 1]
# HasPhrase-2: [1, 0]
# {'HasPhrase': {'iran': 0, 'afghanistan': 1, 'do': 2, 'samaritan': 3}} __ will have to be stored on disk
# {'iran': 0, 'afghanistan': 1, 'do': 0, 'samaritan': 1}

has_phrase_slicer_3 = HasPhrase(phrases=['iran', 'afghanistan', 'do', 'samaritan'])
batch, slices, slice_labels = has_phrase_slicer_3(batch=dataset[:2], keys=['question'])
st.write(pd.DataFrame(slice_labels, columns=has_phrase_slicer.headers))
# HasPhrase-1: [0, 1]
# HasPhrase-2: [1, 0]
# {'iran': 0, 'afghanistan': 1}[1, 0, 0, 1]
