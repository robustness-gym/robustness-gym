Robustness Gym
================================
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/robustness-gym/robustness-gym/CI)
![GitHub](https://img.shields.io/github/license/robustness-gym/robustness-gym)
[![codecov](https://codecov.io/gh/robustness-gym/robustness-gym/branch/main/graph/badge.svg?token=MOLQYUSYQU)](https://codecov.io/gh/robustness-gym/robustness-gym)
[![Documentation Status](https://readthedocs.org/projects/robustnessgym/badge/?version=latest)](https://robustnessgym.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![website](https://img.shields.io/badge/website-live-brightgreen)](https://robustnessgym.com)

Robustness Gym is a evaluation toolkit for natural language processing in Python.

## 

### Installation
```
pip install robustnessgym
```

### Robustness Gym in 5 minutes

#### Datasets that extend Huggingface `datasets`
```python
# robustnessgym.Dataset wraps datasets.Dataset
from robustnessgym import Dataset

# Use Dataset.load_dataset(..) exactly like datasets.load_dataset(..) 
dataset = Dataset.load_dataset('boolq')
dataset = Dataset.load_dataset('boolq', split='train[:10]')
```

#### Cache information
```python
# Get a dataset
from robustnessgym import Dataset
dataset = Dataset.load_dataset('boolq')

# Run the Spacy pipeline
from robustnessgym import Spacy
spacy = Spacy()
# .. on the 'question' column of the dataset
dataset = spacy(batch_or_dataset=dataset, 
                columns=['question'])


# Run the Stanza pipeline
from robustnessgym import Stanza
stanza = Stanza()
# .. on both the question and passage columns of a batch
dataset = stanza(batch_or_dataset=dataset[:32], 
                 columns=['question', 'passage'])

# .. use any of the other built-in operations in Robustness Gym!


# Or, create your own CachedOperation
from robustnessgym import CachedOperation, Identifier
from robustnessgym.core.decorators import singlecolumn

# Write a silly function that operates on a single column of a batch
@singlecolumn
def silly_fn(batch, columns):
    """
    Capitalize text in the specified column of the batch.
    """
    column_name = columns[0]
    assert type(batch[column_name]) == str, "Must apply to text column."
    return [text.capitalize() for text in batch[column_name]] 

# Wrap the silly function in a CachedOperation
silly_op = CachedOperation(apply_fn=silly_fn,
                           identifier=Identifier(_name='SillyOp'))

# Apply it to a dataset
dataset = silly_op(batch_or_dataset=dataset, 
                   columns=['question'])
```


#### Retrieve cached information
```python
from robustnessgym import Spacy, Stanza, CachedOperation

# Take a batch of data
batch = dataset[:32]

# Retrieve the (cached) results of the Spacy CachedOperation 
spacy_information = Spacy.retrieve(batch, columns=['question'])

# Retrieve the tokens returned by the Spacy CachedOperation
tokens = Spacy.retrieve(batch, columns=['question'], proc_fns=Spacy.tokens)

# Retrieve the entities found by the Stanza CachedOperation
entities = Stanza.retrieve(batch, columns=['passage'], proc_fns=Stanza.entities)

# Retrieve the capitalized output of the silly_op
capitalizations = CachedOperation.retrieve(batch,
                                           columns=['question'],
                                           identifier=silly_op.identifier)

# Retrieve it directly using the silly_op
capitalizations = silly_op.retrieve(batch, columns=['question'])

# Retrieve the capitalized output and lower-case it during retrieval
capitalizations = silly_op.retrieve(
    batch,
    columns=['question'],
    proc_fns=lambda decoded_batch: [x.lower() for x in decoded_batch]
)
```

#### Create subpopulations
```python
from robustnessgym import Spacy, ScoreSubpopulation
from robustnessgym.core.decorators import singlecolumn

@singlecolumn
def length(batch, columns):
    """
    Length using cached Spacy tokenization.
    """
    column_name = columns[0]
    # Take advantage of previously cached Spacy informations
    tokens = Spacy.retrieve(batch, columns, proc_fns=Spacy.tokens)[column_name]
    return [len(tokens_) for tokens_ in tokens]

# Create a subpopulation that buckets examples based on length
length_subpopulation = ScoreSubpopulation(intervals=[(0, 10), (10, 20)],
                                          score_fn=length)

dataset, slices, membership = length_subpopulation(dataset, columns=['question'])
# dataset is updated with slice information
# slices is a list of 2 Slice objects
# membership is a matrix of shape (n x 2)
```
