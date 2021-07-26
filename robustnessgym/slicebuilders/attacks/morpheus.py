from typing import Dict, List, Tuple

import numpy as np
from meerkat.tools.lazy_loader import LazyLoader

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import transpose_batch
from robustnessgym.slicebuilders.attack import Attack

morpheus = LazyLoader("morpheus")


class Morpheus(Attack):
    def __init__(
        self,
        dataset: str,
        model: str,
        constrain_pos: bool = True,
        **kwargs,
    ):
        super().__init__(
            identifiers=[
                Identifier(self.__class__.__name__, dataset=dataset, model=model)
            ],
        )

        self.constrain_pos = constrain_pos

        self.dataset = dataset.lower()
        if self.dataset == "mnli":
            self.attack = morpheus.MorpheusHuggingfaceNLI(model)
        elif "squad" in self.dataset:
            is_squad2 = "2" in self.dataset
            self.attack = morpheus.MorpheusHuggingfaceQA(model, squad2=is_squad2)
        elif self.dataset == "cnn_dailymail" or self.dataset == "xsum":
            rouge_type = kwargs.get("rouge_type", "rougeL")
            max_input_tokens = kwargs.get("max_input_tokens", 1024)
            self.attack = morpheus.MorpheusHuggingfaceQA(
                model, rouge_type=rouge_type, max_input_tokens=max_input_tokens
            )
        else:
            raise NotImplementedError

    def apply(
        self,
        skeleton_batches: List[Dict[str, List]],
        slice_membership: np.ndarray,
        batch: Dict[str, List],
        columns: List[str],
        *args,
        **kwargs,
    ) -> Tuple[List[Dict[str, List]], np.ndarray]:

        for i, example in enumerate(transpose_batch(batch)):
            if self.dataset == "mnli":
                # Assume column order is [premise, hypothesis, label]
                prem_col, hypo_col, label_col = columns
                text_label = self.get_NLI_text_label(example[label_col])

                new_prem, new_hypo, predicted_label, _ = self.attack.morph(
                    example[prem_col],
                    example[hypo_col],
                    example[label_col],
                    constrain_pos=self.constrain_pos,
                )
                if predicted_label != text_label:
                    skeleton_batches[0][prem_col][i] = new_prem
                    skeleton_batches[0][hypo_col][i] = new_hypo
                else:
                    slice_membership[i, 0] = 0
            elif "squad" in self.dataset:
                question_col = columns[0]
                # NOTE: assume first element in columns is question_col
                # Ignoring the rest since example['answers'] is another Dict
                question_dict = self.prepare_question_dict(example, question_col)
                new_question, predicted_answer = self.attack.morph(
                    question_dict, example["context"], constrain_pos=self.constrain_pos
                )
                if predicted_answer not in example["answers"]["text"]:
                    skeleton_batches[0][question_col][i] = new_question
                else:
                    slice_membership[i, 0] = 0
            elif self.dataset == "cnn_dailymail" or self.dataset == "xsum":
                # Assume column order is [article_col, summary_col]
                article_col, summary_col = columns
                new_article, predicted_summary, _ = self.attack.morph(
                    example[article_col],
                    example[summary_col],
                    constrain_pos=self.constrain_pos,
                )
                if predicted_summary != example[summary_col]:
                    skeleton_batches[0][article_col][i] = new_article
                else:
                    slice_membership[i, 0] = 0
            else:
                raise NotImplementedError
        return skeleton_batches, slice_membership

    @classmethod
    def prepare_question_dict(cls, example, question_col):
        return {
            "question": example[question_col],
            "answers": [
                {"answer_start": i[0], "text": i[1]}
                for i in zip(
                    example["answers"]["answer_start"], example["answers"]["text"]
                )
            ],
            "is_impossible": len(example["answers"]["text"]) == 0,
        }

    def get_NLI_text_label(self, label: int) -> str:
        hf_labels = ["entailment", "neutral", "contradiction"]
        return hf_labels[label]
