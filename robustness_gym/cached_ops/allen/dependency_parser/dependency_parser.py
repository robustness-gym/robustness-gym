from typing import List

import torch
import allennlp_models.structured_prediction
from allennlp.predictors import Predictor

import robustness_gym.cached_ops.cached_ops


class AllenDependencyParser(robustness_gym.cached_ops.cached_ops.CachedOperation):

    def __init__(self):
        super(AllenDependencyParser, self).__init__(
            identifier='AllenDependencyParser')

        # Set up AllenNLP's depedency parser
        if torch.cuda.is_available():
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
                cuda_device=0
            )
        else:
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
            )

    def apply(self, text_batch) -> List:
        # Apply the dependency parser
        parse_trees = self.predictor.predict_batch_json([
            {'sentence': text} for text in text_batch
        ])

        # Extract the tree from the output of the dependency parser
        return [val['trees'] for val in parse_trees]
