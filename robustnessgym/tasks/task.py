from collections import OrderedDict
from typing import List, Union

from datasets.features import ClassLabel, Sequence, Value

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.tasks.schema import Schema


class Task:
    """Class for tasks in Robustness Gym."""

    dataset_to_task = {}

    def __init__(
        self,
        identifier: Union[str, Identifier] = "GenericTask",
        input_schema: Schema = None,
        output_schema: Schema = None,
        metrics: List[str] = None,
        *args,
        **kwargs,
    ):
        self._identifier = identifier
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._metrics = metrics

    def __repr__(self):
        return (
            f"task: {self._identifier}\n\n"
            f"Input: {str(self._input_schema)}\n\n"
            f"Output: {str(self._output_schema)}"
        )

    @classmethod
    def lookup(cls, dataset: str):
        return cls.dataset_to_task[dataset]

    @classmethod
    def list_datasets(cls):
        return []

    @property
    def identifier(self):
        """Task identifier."""
        return self._identifier

    @property
    def metrics(self):
        """Task metrics."""
        return self._metrics

    @property
    def input_schema(self):
        """Input schema for the task."""
        return self._input_schema

    @property
    def output_schema(self):
        """Output schema for the task."""
        return self._output_schema

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

    def remap_schema(self, dp: DataPanel):
        # Ground the schema to the dp
        input_grounding, reversed_input_grounding = self.input_schema.ground(
            dp.features
        )
        output_grounding, reversed_output_grounding = self.output_schema.ground(
            dp.features
        )

        for col in self.input_schema.columns:
            # Grab the column
            values = dp[input_grounding[col]]
            # Remove it from the dp
            dp.remove_column(input_grounding[col])
            # Add again with the right column name
            dp.add_column(col, values)

        for col in self.output_schema.columns:
            # Grab the column
            values = dp[output_grounding[col]]
            # Remove it from the dp
            dp.remove_column(output_grounding[col])
            # Add again with the right column name
            dp.add_column(col, values)

        return dp

        # # Construct a map_fn that remaps the dataset schema
        # def _map_fn(example):
        #     return tz.merge(
        #         {k: example[input_grounding[k]] for k in self.input_schema.features},
        #         {k: example[output_grounding[k]] for k in self.output_schema.features
        #         },
        #     )
        #
        # return dataset.map(
        #     _map_fn,
        #     remove_columns=list(reversed_input_grounding.keys())
        #     + list(reversed_output_grounding.keys()),
        # )

    def classification(self):
        # TODO(karan): improve the schema inference
        # Check that the only output is a ClassLabel output
        if len(self._output_schema) == 1 and isinstance(
            self._output_schema.features[self._output_schema.columns[0]], ClassLabel
        ):
            return True
        return False


class Generic(Task):
    def __init__(self):
        super(Generic, self).__init__(
            identifier="Generic",
            input_schema=None,
            output_schema=None,
            metrics=None,
        )


class SentimentClassification(Task):
    def __init__(self, identifier, input_schema, output_schema, *args, **kwargs):
        super(SentimentClassification, self).__init__(
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


class BinarySentiment(SentimentClassification):
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
                "f1_micro",
                "f1_macro",
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
