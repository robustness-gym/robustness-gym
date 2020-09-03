from typing import List

import torch
from allennlp.predictors import Predictor

import robustness_gym.cached_ops.cached_ops


class AllenSemanticRoleLabeler(robustness_gym.cached_ops.cached_ops.CachedOperation):

    def __init__(self):
        super(AllenSemanticRoleLabeler, self).__init__(
            identifier='AllenSemanticRoleLabeler')

        # Set up AllenNLP's constituency parser
        if torch.cuda.is_available():
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz",
                cuda_device=0
            )
        else:
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz",
            )

    def apply(self, text_batch) -> List:
        # Apply the semantic role labeler
        role_labels = self.predictor.predict_batch_json([
            {'sentence': text} for text in text_batch
        ])

        # Extract the tree from the output of the constituency parser
        return [val['trees'] for val in role_labels]

