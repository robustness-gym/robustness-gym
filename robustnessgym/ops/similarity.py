from __future__ import annotations

from typing import List, Sequence

import cytoolz as tz
import numpy as np
from datasets import load_metric

from robustnessgym.core.operation import Operation, lookup
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.core.tools import transpose_batch
from robustnessgym.ops.spacy import SpacyOp


class SentenceSimilarityMatrixOp(Operation):
    def __init__(self):
        super(SentenceSimilarityMatrixOp, self).__init__()

    def similarity(
        self,
        batch_sentences_1: List[List[str]],
        batch_sentences_2: List[List[str]],
    ) -> List:
        raise NotImplementedError("Must implement a similarity computation.")

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> tuple:
        """

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            **kwargs: optional keyword arguments

        Returns:
            Tuple with single output
        """

        assert len(columns) == 2, "Exactly two columns required."

        # Lookup the sentences in the given columns
        [lookup(dp, SpacyOp, [col]) for col in columns]
        sentences = SpacyOp.retrieve(
            batch=dp,
            columns=[[col] for col in columns],
            proc_fns=SpacyOp.sentences,
        )

        return self.similarity(*[sentences[col] for col in columns])


class DocumentSimilarityScore(Operation):
    def __init__(self):
        super(DocumentSimilarityScore, self).__init__()
        self.metric = load_metric("rouge")

    def similarity(self, batch_doc_1: List[str], batch_doc_2: List[str]):
        raise NotImplementedError("Must implement a similarity computation.")

    def apply(self, dp, columns, **kwargs):
        assert len(columns) == 2
        return self.similarity(*[dp[col] for col in columns])


class RougeScore(DocumentSimilarityScore):
    def __init__(self):
        super(RougeScore, self).__init__()
        self.metric = load_metric("rouge")

    def similarity(self, batch_doc_1: List[str], batch_doc_2: List[str]):
        # Compute the scores between every pair of documents
        scores = self.metric.compute(
            predictions=batch_doc_1, references=batch_doc_2, use_agregator=False
        )

        # Transpose the batch of scores
        scores = [
            tz.valmap(
                lambda v: {
                    m: getattr(v, m) for m in ["precision", "recall", "fmeasure"]
                },
                example,
            )
            for example in transpose_batch(scores)
        ]

        return scores

    @classmethod
    def select(
        cls, decoded_batch: List, metric: Sequence[str] = ("rouge1", "fmeasure")
    ):
        if len(metric) == 1:
            return [scores[metric[0]] for scores in decoded_batch]
        elif len(metric) == 2:
            return [scores[metric[0]][metric[1]] for scores in decoded_batch]
        else:
            raise ValueError(f"metric {metric} must be a sequence of length <= 2.")


class RougeMatrix(SentenceSimilarityMatrixOp):
    def __init__(self):
        super(RougeMatrix, self).__init__()
        self.metric = load_metric("rouge")

    def similarity(
        self, batch_sentences_1: List[List[str]], batch_sentences_2: List[List[str]]
    ):
        batch_similarity = []
        for sents_1, sents_2 in zip(batch_sentences_1, batch_sentences_2):
            # Compute the scores between every pair of sentences
            scores = self.metric.compute(
                predictions=np.repeat(sents_1, len(sents_2)),
                references=sents_2 * len(sents_1),
                use_agregator=False,
            )

            # Organize all the scores into a similarity matrix for each metric
            similarity_mat = {
                k: {
                    m: np.array([getattr(e, m) for e in v])
                    .reshape(len(sents_1), len(sents_2))
                    .tolist()
                    for m in ["precision", "recall", "fmeasure"]
                }
                for k, v in scores.items()
            }

            batch_similarity.append(similarity_mat)

        return batch_similarity

    @classmethod
    def select(
        cls, decoded_batch: List, metric: Sequence[str] = ("rouge1", "fmeasure")
    ):
        if len(metric) == 1:
            return [
                tz.valmap(np.array, matrices[metric[0]]) for matrices in decoded_batch
            ]
        elif len(metric) == 2:
            return [
                np.array(matrices[metric[0]][metric[1]]) for matrices in decoded_batch
            ]
        else:
            raise ValueError(f"metric {metric} must be a sequence of length <= 2.")
