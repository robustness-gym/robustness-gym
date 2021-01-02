from collections import OrderedDict
from typing import List

import cytoolz as tz
from datasets.features import ClassLabel, Sequence, Value

from robustnessgym.core.dataset import Dataset
from robustnessgym.tasks.schema import Schema


class Task:
    dataset_to_task = {}

    def __init__(
        self,
        identifier,
        input_schema: Schema,
        output_schema: Schema,
        metrics: List[str],
        *args,
        **kwargs,
    ):
        self.identifier = identifier
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.metrics = metrics

    @classmethod
    def lookup(cls, dataset: str):
        return cls.dataset_to_task[dataset]

    @classmethod
    def list_datasets(cls):
        return []

    # @classmethod
    # def from_identifier(cls, identifier):
    #     return getattr()

    @classmethod
    def create(cls, task: str):
        # TODO(karan): figure out how to getattr this
        if task == "TernaryNaturalLanguageInference":
            return TernaryNaturalLanguageInference()
        else:
            raise NotImplementedError

    def remap_schema(self, dataset: Dataset):
        # Ground the schema to the dataset
        input_grounding, reversed_input_grounding = self.input_schema.ground(
            dataset.features
        )
        output_grounding, reversed_output_grounding = self.output_schema.ground(
            dataset.features
        )

        # Construct a map_fn that remaps the dataset schema
        def map_fn(example):
            return tz.merge(
                {k: example[input_grounding[k]] for k in self.input_schema.features},
                {k: example[output_grounding[k]] for k in self.output_schema.features},
            )

        return dataset.map(
            map_fn,
            remove_columns=list(reversed_input_grounding.keys())
            + list(reversed_output_grounding.keys()),
        )

    def classification(self):
        # TODO(karan): improve the schema inference
        # Check that the only output is a ClassLabel output
        if len(self.output_schema) == 1 and isinstance(
            self.output_schema.features[self.output_schema.keys()[0]], ClassLabel
        ):
            return True
        return False

    def __repr__(self):
        return (
            f"task: {self.identifier}\n\nInput{str(self.input_schema)}\n\nOutput"
            f"{str(self.output_schema)}"
        )


# class ClassificationMixin:
#
#     def __init__(self,
#                  num_classes: int = None,
#                  *args,
#                  **kwargs):
#         super(ClassificationMixin, self).__init__(*args, **kwargs)
#
#         self.output_schema = None


class Sentiment(Task):
    def __init__(self, identifier, input_schema, output_schema, *args, **kwargs):
        super(Sentiment, self).__init__(
            identifier=identifier,
            input_schema=input_schema,
            output_schema=output_schema,
            metrics=[
                "accuracy",
                "f1",
                "class_dist",
                "pred_dist"
                # TODO(karan): calibration, other metrics
            ],
            *args,
            **kwargs,
        )


class BinarySentiment(Sentiment):
    def __init__(self):
        super(BinarySentiment, self).__init__(
            num_classes=2,
            input_schema=Schema(
                features=OrderedDict(
                    [
                        ("text", Value(dtype="string")),
                    ]
                ),
                grounding_candidates={
                    "text": {"text", "sentence"},
                },
            ),
            output_schema=Schema(
                features=OrderedDict(
                    [
                        ("label", ClassLabel(names=["negative", "positive"])),
                    ]
                ),
                grounding_candidates={
                    "label": {"label"},
                },
            ),
            identifier=self.__class__.__name__,
        )

    @classmethod
    def list_datasets(cls):
        return [
            "imdb",
        ]


class Summarization(Task):
    def __init__(self):
        super(Summarization, self).__init__(
            identifier=self.__class__.__name__,
            input_schema=Schema(
                features=OrderedDict([("text", Value(dtype="string"))]),
                grounding_candidates={
                    "text": {"article", "document"},
                },
            ),
            output_schema=Schema(
                features=OrderedDict([("summary", Value(dtype="string"))]),
                grounding_candidates={
                    "summary": {"highlights", "summary"},
                },
            ),
            metrics=[
                # blah,
                # TODO(karan): calibration, other metrics
                "rouge1",
                "rouge2",
                "rougeLsum",
            ],
        )

    @classmethod
    def list_datasets(cls):
        return [
            "cnn_dailymail",
        ]


class NaturalLanguageInference(Task):
    def __init__(self, identifier, input_schema, output_schema, *args, **kwargs):
        super(NaturalLanguageInference, self).__init__(
            identifier=identifier,
            input_schema=input_schema,
            output_schema=output_schema,
            metrics=[
                "accuracy",
                "f1",
                "class_dist",
                "pred_dist"
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
                features=OrderedDict(
                    [
                        ("premise", Value(dtype="string")),
                        ("hypothesis", Value(dtype="string")),
                    ]
                ),
                grounding_candidates={
                    "premise": {"premise", "sentence1"},
                    "hypothesis": {"hypothesis", "sentence2"},
                },
            ),
            output_schema=Schema(
                features=OrderedDict(
                    [
                        ("label", ClassLabel(names=["entailment", "non entailment"])),
                    ]
                ),
                grounding_candidates={
                    "label": {"label"},
                },
            ),
            identifier=self.__class__.__name__,
        )

    @classmethod
    def list_datasets(cls):
        return []


class TernaryNaturalLanguageInference(NaturalLanguageInference):
    def __init__(self):
        super(TernaryNaturalLanguageInference, self).__init__(
            num_classes=3,
            input_schema=Schema(
                features=OrderedDict(
                    [
                        ("premise", Value(dtype="string")),
                        ("hypothesis", Value(dtype="string")),
                    ]
                ),
                grounding_candidates={
                    "premise": {"premise", "sentence1"},
                    "hypothesis": {"hypothesis", "sentence2"},
                },
            ),
            output_schema=Schema(
                features=OrderedDict(
                    [
                        (
                            "label",
                            ClassLabel(
                                names=["entailment", "neutral", "contradiction"]
                            ),
                        ),
                    ]
                ),
                grounding_candidates={
                    "label": {"label"},
                },
            ),
            identifier=self.__class__.__name__,
        )

    def datasets(self):
        return {
            "snli",
        }


class QuestionAnswering(Task):
    def __init__(self, identifier, input_schema, output_schema, *args, **kwargs):
        super(QuestionAnswering, self).__init__(
            identifier=identifier,
            input_schema=input_schema,
            output_schema=output_schema,
            metrics=[
                "em",
                "f1",
                # TODO(karan): calibration, other metrics
            ],
            *args,
            **kwargs,
        )


class ExtractiveQuestionAnswering(Task):
    def __init__(self):
        super(ExtractiveQuestionAnswering, self).__init__(
            input_schema=Schema(
                features=OrderedDict(
                    [
                        ("context", Value(dtype="string")),
                        ("question", Value(dtype="string")),
                    ]
                ),
                grounding_candidates={
                    "context": {"context"},
                    "question": {"question"},
                },
            ),
            output_schema=Schema(
                features=OrderedDict(
                    [
                        (
                            "answers",
                            Sequence(
                                feature={
                                    "text": Value(dtype="string", id=None),
                                    "answer_start": Value(dtype="int32", id=None),
                                },
                                length=-1,
                            ),
                        ),
                    ]
                ),
                grounding_candidates={
                    "answers": {
                        "answers",
                    },
                },
            ),
            metrics=[
                "em",
                "f1",
            ],
            identifier=self.__class__.__name__,
        )


# class ExtractiveQuestionAnswering(Task):
#
#     def __init__(self):
#         super(ExtractiveQuestionAnswering, self).__init__(
#             input_schema=Schema(
#                 features=OrderedDict([
#                     ('context', Value(dtype='string')),
#                     ('question', Value(dtype='string')),
#                 ]),
#                 grounding_candidates={
#                     'context': {'context'},
#                     'question': {'question'},
#                 },
#             ),
#             output_schema=Schema(
#                 features=OrderedDict([
#                     ('answer', Sequence(Value(dtype='string'), length=-1)),
#                     ('start', Sequence(Value(dtype='int64'), length=-1)),
#                     ('end', Sequence(Value(dtype='int64'), length=-1)),
#                 ]),
#                 grounding_candidates={
#                     'answer': {
#                         ('answers', 'text'),
#                     },
#                     'start': {
#                         ('answers', 'answer_start')
#                     },
#                     'end': {
#                         lambda answer, start: [idx + len(answer) for idx in start],
#                     },
#                 }
#             ),
#             metrics=[
#                 'em',
#                 'f1',
#             ],
#             identifier=self.__class__.__name__,
#         )

# Evaluation Hierarchy
# --------------------
# (generic task, model) ### QuestionAnswering/NLI
# (narrow task, model) ### MultiHopQuestionAnswering/BinaryNLI
# (dataset, model) ### Particular Dataset/QNLI
