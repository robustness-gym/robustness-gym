from typing import List

import torch
from allennlp.predictors import Predictor

from robustness_gym.cached_ops.cached_ops import CachedOperation
from robustness_gym.identifier import Identifier


class AllenConstituencyParser(CachedOperation):

    def __init__(self):
        super(AllenConstituencyParser, self).__init__(
            identifier=Identifier(name=self.__class__.__name__),
        )

        # Set up AllenNLP's constituency parser
        if torch.cuda.is_available():
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
                cuda_device=0
            )
        else:
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
            )

    def apply(self, text_batch) -> List:
        # Apply the constituency parser
        parse_trees = self.predictor.predict_batch_json([
            {'sentence': text} for text in text_batch
        ])

        # Extract the tree from the output of the constituency parser
        return [val['trees'] for val in parse_trees]
