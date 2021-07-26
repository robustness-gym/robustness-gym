<div align="center">
    <img src="docs/logo.png" height=100 alt="RG logo"/>
    <h1 style="font-family: 'IBM Plex Sans'">Robustness Gym</h1>
</div>

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/robustness-gym/robustness-gym/CI)
![GitHub](https://img.shields.io/github/license/robustness-gym/robustness-gym)
[![Documentation Status](https://readthedocs.org/projects/robustnessgym/badge/?version=latest)](https://robustnessgym.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![website](https://img.shields.io/badge/website-live-brightgreen)](https://robustnessgym.com)

[comment]: <> ([![codecov]&#40;https://codecov.io/gh/robustness-gym/robustness-gym/branch/main/graph/badge.svg?token=MOLQYUSYQU&#41;]&#40;https://codecov.io/gh/robustness-gym/robustness-gym&#41;)

Robustness Gym is a Python evaluation toolkit for machine learning models. 

[**Getting Started**](#getting-started)
| [**What is Robustness Gym?**](#what-is-robustness-gym)
| [**Docs**](https://robustnessgym.readthedocs.io/en/latest/index.html)
| [**Contributing**](CONTRIBUTING.md)
| [**About**](#about)


### Getting started
```
pip install robustnessgym
```
> Note: some parts of Robustness Gym rely on optional dependencies. 
> If you know which optional dependencies you'd like to install, 
> you can do so using something like `pip install robustnessgym[dev,text]` instead. 
> See `setup.py` for a full list of optional dependencies.

### What is Robustness Gym?
Robustness Gym is being developed to address challenges in evaluating machine 
learning models today, with tools to evaluate and visualize the quality of machine 
learning models. 

Along with [Meerkat](https://github.com/robustness-gym/mosaic), 
we make it easy for you to load in any kind of data 
(text, images, videos, time-series) and quickly evaluate how well your models are 
performing.

#### Load data into a Meerkat `DataPanel`
```python
from robustnessgym import DataPanel

# Any Huggingface dataset
dp = DataPanel.load_huggingface('boolq')

# Custom datasets
dp = DataPanel.from_csv(...)
dp = DataPanel.from_pandas(...)
dp = DataPanel.from_jsonl(...)
dp = DataPanel.from_feather(...)

# Coming soon: any WILDS dataset
# from meerkat.contrib.wilds import get_wilds_datapanel
# dp = get_wilds_datapanel("fmow", root_dir="/datasets/", split="test")
```

### Run common workflows

#### Spacy
```python
from robustnessgym import DataPanel, lookup
from robustnessgym.ops import SpacyOp

dp = DataPanel.load_huggingface('boolq')

# Run the Spacy pipeline on the 'question' column of the dataset
spacy = SpacyOp()
dp = spacy(dp=dp, columns=['question'])
# adds a new column that is auto-named
# "SpacyOp(lang=en_core_web_sm, neuralcoref=False, columns=['passage'])"

# Grab the Spacy column from the DataPanel using the lookup
spacy_column = lookup(dp, spacy, ['question'])
```

#### Stanza
```python
from robustnessgym import DataPanel, lookup
from robustnessgym.ops import StanzaOp

dp = DataPanel.load_huggingface('boolq')

# Run the Stanza pipeline on the 'question' column of the dataset
stanza = StanzaOp()
dp = stanza(dp=dp, columns=['question'])
# adds a new column that is auto-named "StanzaOp(columns=['question'])"

# Grab the Stanza column from the DataPanel using the lookup
stanza_column = lookup(dp, stanza, ['question'])
```

#### Custom Operation (Single Output)
```python
# Or, create your own Operation
from robustnessgym import DataPanel, Operation, Id, lookup

dp = DataPanel.load_huggingface('boolq')

# A function that capitalizes text
def capitalize(batch: DataPanel, columns: list):
    return [text.capitalize() for text in batch[columns[0]]]

# Wrap in an Operation: `process_batch_fn` accepts functions that have
# exactly 2 arguments: batch and columns, and returns a tuple of outputs
op = Operation(
    identifier=Id('CapitalizeOp'),
    process_batch_fn=capitalize,
)

# Apply to a DataPanel
dp = op(dp=dp, columns=['question'])

# Look it up when you need it
capitalized_text = lookup(dp, op, ['question'])
```

#### Custom Operation (Multiple Outputs)
```python
from robustnessgym import DataPanel, Operation, Id, lookup

dp = DataPanel.load_huggingface('boolq')

# A function that capitalizes and upper-cases text: this will
# be used to add two columns to the DataPanel
def capitalize_and_upper(batch: DataPanel, columns: list):
    return [text.capitalize() for text in batch[columns[0]]], \
           [text.upper() for text in batch[columns[0]]]

# Wrap in an Operation: `process_batch_fn` accepts functions that have
# exactly 2 arguments: batch and columns, and returns a tuple of outputs
op = Operation(
    identifier=Id('ProcessingOp'),
    output_names=['capitalize', 'upper'],  # tell the Operation the name of the two outputs
    process_batch_fn=capitalize_and_upper,
)

# Apply to a DataPanel
dp = op(dp=dp, columns=['question'])

# Look them up when you need them
capitalized_text = lookup(dp, op, ['question'], 'capitalize')
upper_text = lookup(dp, op, ['question'], 'upper')
```


### Create Evaluations


#### Out-of-the-box Subpopulations
```python
from robustnessgym import DataPanel
from robustnessgym import LexicalOverlapSubpopulation

dp = DataPanel.load_huggingface('boolq')

# Create a subpopulation that buckets examples based on length
lexo_sp = LexicalOverlapSubpopulation(intervals=[(0., 0.1), (0.1, 0.2)])

slices, membership = lexo_sp(dp=dp, columns=['question'])
# `slices` is a list of 2 DataPanel objects
# `membership` is a matrix of shape (n x 2)
```

#### Custom Subpopulation
```python
from robustnessgym import DataPanel, ScoreSubpopulation, lookup
from robustnessgym.ops import SpacyOp

dp = DataPanel.load_huggingface('boolq')

def length(batch: DataPanel, columns: list):
    try:
        # Take advantage of previously stored Spacy information
        return [len(doc) for doc in lookup(batch, SpacyOp, columns)] 
    except AttributeError:
        # If unavailable, fall back to splitting text
        return [len(text.split()) for text in batch[columns[0]]]
    
# Create a subpopulation that buckets examples based on length
length_sp = ScoreSubpopulation(intervals=[(0, 10), (10, 20)], score_fn=length)

slices, membership = length_sp(dp=dp, columns=['question'])
# `slices` is a list of 2 DataPanel objects
# `membership` is a matrix of shape (n x 2)
```


### About
 You can read more about the ideas underlying Robustness Gym in our 
paper on [arXiv](https://arxiv.org/pdf/2101.04840.pdf).

The Robustness Gym project began as a collaboration between [Stanford Hazy
 Research](https://hazyresearch.stanford.edu), [Salesforce Research](https://einstein.ai
 ) and [UNC Chapel-Hill](http://murgelab.cs.unc.edu/). We also have a
   [website](https://robustnessgym.com).

If you use Robustness Gym in your work, please use the following BibTeX entry,
```
@inproceedings{goel-etal-2021-robustness,
    title = "Robustness Gym: Unifying the {NLP} Evaluation Landscape",
    author = "Goel, Karan  and
      Rajani, Nazneen Fatema  and
      Vig, Jesse  and
      Taschdjian, Zachary  and
      Bansal, Mohit  and
      R{\'e}, Christopher",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-demos.6",
    pages = "42--55",
}
```