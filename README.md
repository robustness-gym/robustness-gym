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

### Using Robustness Gym
```python
import robustnessgym as rg

# Load any dataset
sst = rg.DataPanel.from_huggingface('sst', split='validation')

# Load any model
sst_model = rg.HuggingfaceModel('distilbert-base-uncased-finetuned-sst-2-english', is_classifier=True)

# Generate predictions for first 2 examples in dataset using "sentence" column as input
predictions = sst_model.predict_batch(sst[:2], ['sentence'])

# Run inference on an entire dataset & store the predictions in the dataset
sst = sst.update(lambda x: sst_model.predict_batch(x, ['sentence']), batch_size=4, is_batched_fn=True, pbar=True)

# Create a DevBench, which will contain slices to evaluate
sst_db = rg.DevBench()

# Add slices of data; to begin with let's add the full dataset
# Slices are just datasets that you can track performance on
sst_db.add_slices([sst])

# Let's add another slice by filtering examples containing negation words
sst_db(rg.HasNegation(), sst, ['sentence'])

# Add any metrics you like
sst_db.add_aggregators({
    # Map from model name to dictionary of metrics
    'distilbert-base-uncased-finetuned-sst-2-english': {
        # This function uses the predictions we stored earlier to calculate accuracy
        'accuracy': lambda dp: (dp['label'].round() == dp['pred'].numpy()).mean()
    }
})

# Create a report
report = sst_db.create_report()

# Visualize: requires installing plotly support in Jupyter, generally works better in Jupyter notebooks (rather than Jupyter Lab)
report.figure()

# Alternatively, save report to file
report.figure().write_image('sst_db_report.png', engine='kaleido')

```

#### Applying Built-in Subpopulations
```python

# Create a slicebuilder that creates subpopulations based on length, in this case the bottom and top 10 percentile.
length_sb = rg.NumTokensSubpopulation(intervals=[("0%", "10%"), ("90%", "100%")])

slices, membership = length_sb(dp=sst, columns=['sentence'])
# `slices` is a list of 2 DataPanel objects
# `membership` is a matrix of shape (n x 2)
for sl in slices:
    print(sl.identifier)
```

#### Creating Custom Subpopulations
```python

def length(batch: rg.DataPanel, columns: list):
    return [len(text.split()) for text in batch[columns[0]]]
    
# Create a subpopulation that buckets examples based on length
length_sp = rg.ScoreSubpopulation(intervals=[(0, 10), (10, 20)], score_fn=length)

slices, membership = length_sp(dp=sst, columns=['sentence'])
for sl in slices:
    print(sl.identifier)
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