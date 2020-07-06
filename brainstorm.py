for sf in slicing_functions:
    for example in dataset:
        label = apply(sf, example)
        # do stuff

# How should people write slices?

    # We only support a slicing function that takes < k seconds per example.
    # Give them the dataset, have them run it, check if it's fast otherwise make it more efficient

    # Can we get people to upload (jsonl?) files that contain information: {'length': 4}, {'POS': blah}: other parameters

# What are the generic slicing functions?
    # e.g. HasPhrase

    # Look for words, phrases
    # Look for any word or phrase in a list
    # Look for either word_1 or/and word_2 or/and word_3
    # Look for things that are some length
    # Slices that depend on running some other "thing" (POS-tagger, Constitutency Parser, NER, SRL)
        # We could just cache
        # The "example" is something that contains text_a, text_b, but also these other properties

    # Slices could run other slices inside (composition)
    # Some slices can only run on pairs (lexical overlap)


# What should appear on the dashboard? What are the views?
    # Being able to control whether the slicing function runs on text_a, text_b or both
    # Size of the slices (bar to control this size)
    # Another view could become specializing a slice to a class


# Table

Slice
-->
SliceWithPredictions(pd.DataFrame([examples, model_i_predictions]))  # model_i_predictions is a list of probs


# 1 Model Many Slices
# If we had a list of [SlicewithPredictions], then we could generate the visualizations?
# Model card style thing, we want this!

def visualize(slice_with_predictions):
    for sp in slice_with_predictions:

        if len(sp) < 10:
            continue

        # Blah blah


# -------------------------------------------------------

# Categorization for Slicing Functions
    # There are a lot of slices, we don't want to overwhelm people

    # How to make the radar chart?

    # Syntactic


# Visualizations

    # Select the type of Slice
    # ['augmentation', 'adv_attack', 'eval_set', 'slicing functions', 'dataset']

    # Views

    # Summary of Robustness Page (depends on Model)
        # Summarize what this model does well on, what it fails on
        # Radar plot
        # Aggregate metrics

    # Only Slice (no Model)

        # Similar to huggingface nlp viewer

    # 1 Model 1 Slice <----- look at examples

        # Look at examples (randomly; top 5) inside the slice (visualize probs over labels per example)

    # 1 Model Many Slices <----!! this definnitely

        # Radar Plot View


    # Many Models 1 Slice <---- not this

    # Many Models Many Slices <---- this only for a few (<= 3 models)

        # Radar Plot View


# predictions should be generic, but we only care about distribution over classes


class Example:

    def __init__(self):
        self.guid = None
        self.text_a = None
        self.text_b = None
        self.label = None

        self.properties = {
            'spacy_nlp': None,  # POS, NER, ...?
            'allen_nlp': None,
            # ...
        }


for slice in slices:
    if slice.info.type == 'augmentation':
    # then do something
    else:
        continue


class Slice:
    examples: Sequence[Example]

    def __init__(self):
        # A slice contains a list of examples
        self.examples = []

        # A slice contains information about how it was derived
        self.info = {

            'type': [],  # ['augmentation', 'adv_attack', 'eval_set', 'sfs', 'dataset'],
            'tasks': [],  # ['NLI'],

            'split': None,  # ['train', 'val', 'test'] --> val is most likely for our library

            # Dependencies is
            'dependencies': {},  # dependence find a better word

        }

    def add_examples(self, examples: Sequence[Example]) -> None:
        self.examples.append(examples)


class SlicingFunction:
    groups = []

    def __init__(self):
        pass

    def label(self, text_a, text_b=None, return_dict=False):
        # Get the raw predictions first
        predictions = self.predictor(text_a, text_b)

        if return_dict:
            return dict(zip(self.groups, predictions))

        return predictions

    def predictor(self, text_a, text_b):
        return []


class Augmentation:
    pass


class TextAttack:
    # Metadata
    version: int

    def __init__(self):

    @property
    def version(self):
        return '0.1.5'

    def create_slice(self, params):
        # Call textattack (do magic with params)

        slice = Slice(new_examples, dependence={
            'library': 'textattack',
            'recipe': 'GeneticAlgorithmAlzantot2018',
        })

        return slice

        pass