from collections import OrderedDict
from typing import *

import cytoolz as tz
from nlp.features import ClassLabel, Value, FeatureType

from robustness_gym.dataset import Dataset


class Space:

    def __init__(self):
        pass


class Schema:

    def __init__(self,
                 features: OrderedDict,
                 grounding_candidates: Dict[str, Collection]):
        self.features = features
        self.grounding_candidates = grounding_candidates
        self.reversed_grounding_candidates = {v: k for k, values in self.grounding_candidates.items() for v in values}

    def ground(self, features: Dict[str, FeatureType]):
        # Figure out the (reversed) grounding: map columns in the dataset to keys in the schema
        reversed_grounding = tz.keyfilter(lambda k: k in features, self.reversed_grounding_candidates)

        # Construct the grounding by reversing
        grounding = {v: k for k, v in reversed_grounding.items()}

        # Assert that the grounding covers the entire schema
        assert len(self.features) == len(grounding), "Grounding failed."

        # Assert that the grounded schema has the right types
        # TODO(karan): if not, add code to automatically rejig the dataset in map_fn
        for key in self.features:
            assert self.features[key] == features[grounding[key]]

        return grounding, reversed_grounding

    def __repr__(self):
        features = "\n\t".join([f"{k}: {v}" for k, v in self.features.items()])
        return f"Schema(\n\t{features}\n)"

    def __len__(self):
        return len(self.features)

    def keys(self):
        return list(self.features.keys())


class Task:
    dataset_to_task = {}

    def __init__(self,
                 identifier,
                 input_schema,
                 output_schema,
                 metrics,
                 *args,
                 **kwargs
                 ):
        self.identifier = identifier
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.metrics = metrics

    @classmethod
    def lookup(cls, dataset: str):
        return cls.dataset_to_task[dataset]

    # @classmethod
    # def from_identifier(cls, identifier):
    #     return getattr()

    @classmethod
    def create(cls, task: str):
        # TODO(karan): figure out how to getattr this
        if task == 'TernaryNaturalLanguageInference':
            return TernaryNaturalLanguageInference()
        else:
            raise NotImplementedError

    def remap_schema(self, dataset: Dataset):
        # Ground the schema to the dataset
        input_grounding, reversed_input_grounding = self.input_schema.ground(dataset.features)
        output_grounding, reversed_output_grounding = self.output_schema.ground(dataset.features)

        # Construct a map_fn that remaps the dataset schema
        def map_fn(example):
            return tz.merge({k: example[input_grounding[k]] for k in self.input_schema.features},
                            {k: example[output_grounding[k]] for k in self.output_schema.features})

        return dataset.map(
            map_fn,
            remove_columns=list(reversed_input_grounding.keys()) + list(reversed_output_grounding.keys())
        )

    def classification(self):
        # TODO(karan): improve the schema inference
        # Check that the only output is a ClassLabel output
        if len(self.output_schema) == 1 and isinstance(self.output_schema.features[self.output_schema.keys()[0]],
                                                       ClassLabel):
            return True
        return False

    def __repr__(self):
        return f"task: {self.identifier}\n\nInput{str(self.input_schema)}\n\nOutput{str(self.output_schema)}"


class ClassificationMixin:

    def __init__(self,
                 num_classes,
                 *args,
                 **kwargs):
        super(ClassificationMixin, self).__init__(*args, **kwargs)

        self.output_schema = None


class NaturalLanguageInference(Task,
                               ClassificationMixin):

    def __init__(self,
                 identifier,
                 input_schema,
                 output_schema,
                 *args,
                 **kwargs):
        super(NaturalLanguageInference, self).__init__(
            identifier=identifier,
            input_schema=input_schema,
            output_schema=output_schema,
            metrics=[
                'accuracy',
                'f1',
                'class_dist',
                'pred_dist'
                # TODO(karan): calibration, other metrics
            ],
            *args,
            **kwargs,
        )


class BinaryNaturalLanguageInference(NaturalLanguageInference):

    def __init__(self):
        super(BinaryNaturalLanguageInference, self).__init__(
            num_classes=2,
            input_schema=Schema(
                features=OrderedDict([
                    ('premise', Value(dtype='string')),
                    ('hypothesis', Value(dtype='string')),
                ]),
                grounding_candidates={
                    'premise': {'premise', 'sentence1'},
                    'hypothesis': {'hypothesis', 'sentence2'},
                }
            ),
            output_schema=Schema(
                features=OrderedDict([
                    ('label', ClassLabel(names=['entailment', 'non entailment'])),
                ]),
                grounding_candidates={
                    'label': {'label'},
                }
            ),
            identifier=self.__class__.__name__,
        )


class TernaryNaturalLanguageInference(NaturalLanguageInference):

    def __init__(self):
        super(TernaryNaturalLanguageInference, self).__init__(
            num_classes=3,
            input_schema=Schema(
                features=OrderedDict([
                    ('premise', Value(dtype='string')),
                    ('hypothesis', Value(dtype='string')),
                ]),
                grounding_candidates={
                    'premise': {'premise', 'sentence1'},
                    'hypothesis': {'hypothesis', 'sentence2'},
                }
            ),
            output_schema=Schema(
                features=OrderedDict([
                    ('label', ClassLabel(names=['entailment', 'neutral', 'contradiction'])),
                ]),
                grounding_candidates={
                    'label': {'label'},
                }
            ),
            identifier=self.__class__.__name__,
        )

    def datasets(self):
        return {
            'snli',
        }


class QuestionAnswering(Task):

    def __init__(self):
        super(QuestionAnswering, self).__init__(
            identifier=self.__class__.__name__,
            input_schema=Tuple[str, str],
            output_schema=int,
            metrics=[
                # blah,
                # TODO(karan): calibration, other metrics
            ],

        )

# Evaluation Hierarchy
# --------------------
# (generic task, model) ### QuestionAnswering/NLI
# (narrow task, model) ### MultiHopQuestionAnswering/BinaryNLI
# (dataset, model) ### Particular Dataset/QNLI

class Summarization(Task):

    def __init__(self):
        super(QuestionAnswering, self).__init__(
            identifier=self.__class__.__name__,
            input_schema=Tuple[str, str],
            output_schema=int,
            metrics=[
                # blah,
                # TODO(karan): calibration, other metrics
            ],

        )