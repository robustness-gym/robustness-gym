import os
import time
import torch
import logging

from typing import Optional, Union, List, Dict
from transformers import (
    GlueDataset,
    GlueDataTrainingArguments,
    PreTrainedTokenizer,
    glue_convert_examples_to_features,
    InputExample,
    DataProcessor,
    # Used in label-flipping hacks
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    BartTokenizer,
    BartTokenizerFast)

from transformers.data.datasets.glue import (
    Split,
    FileLock)
from transformers.data.processors.glue import (
    MnliProcessor,
    MnliMismatchedProcessor)
from transformers.data.metrics import simple_accuracy

logger = logging.getLogger(__name__)


class CustomGlueDataset(GlueDataset):
    """Customized GlueData with changes:

    1. Changed the `glue_processors` and `glue_output_modes` to customized ones.
    """

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm", "mnli-2", "mnli-2-mm", "hans"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
            BartTokenizer,
            BartTokenizerFast,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


class TwoLabelMnliProcessor(MnliProcessor):

    def get_labels(self) -> List[str]:
        """See base class."""
        return ["non_entailment", "entailment"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[InputExample]:
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else self._preprocess_label(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _preprocess_label(self, label: str) -> str:
        if label not in ["contradiction", "entailment", "neutral"]:
            raise ValueError(f"Label {label} not recognized.")

        if label in ["contradiction", "neutral"]:
            return "non_entailment"
        else:
            return "entailment"


class TwoLabelMnliMismatchedProcessor(TwoLabelMnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class HansProcessor(DataProcessor):
    """Processor for the HANS data set."""

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "heuristics_train_set.txt")), "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "heuristics_evaluation_set.txt")), "dev")

    def get_labels(self) -> List[str]:
        """See base class."""
        return ["non_entailment", "entailment"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[5]
            text_b = line[6]
            label = self._preprocess_label(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _preprocess_label(self, label: str) -> str:
        if label not in ["non-entailment", "entailment"]:
            raise ValueError(f"Label {label} not recognized.")

        if label in ["non-entailment"]:
            return "non_entailment"
        else:
            return "entailment"


def glue_compute_metrics(task_name: str, preds: List, labels: List) -> Dict[str, float]:
    assert len(preds) == len(labels)
    if task_name not in glue_processors.keys():
        raise ValueError(f"Unrecognized {task_name}")

    return {"acc": simple_accuracy(preds, labels)}


glue_tasks_num_labels = {
    "mnli": 3,
    "mnli-2": 2,
    "hans": 2,
}

glue_processors = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mnli-2": TwoLabelMnliProcessor,
    "mnli-2-mm": TwoLabelMnliMismatchedProcessor,
    "hans": HansProcessor,
}

glue_output_modes = {
    "mnli": "classification",
    "mnli-mm": "classification",
    "mnli-2": "classification",
    "mnli-2-mm": "classification",
    "hans": "classification",
}
