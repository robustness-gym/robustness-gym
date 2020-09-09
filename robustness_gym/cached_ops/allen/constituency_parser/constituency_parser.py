from typing import List

from robustness_gym.cached_ops.allen.allen_predictor import AllenPredictor


class AllenConstituencyParser(AllenPredictor):

    def __init__(self,
                 device: str = None):
        super(AllenConstituencyParser, self).__init__(
            path="https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
            device=device,
        )

    def apply(self, text_batch) -> List:
        # Apply the constituency parser
        parse_trees = self.predictor.predict_batch_json([
            {'sentence': text} for text in text_batch
        ])

        # Extract the tree from the output of the constituency parser
        return [val['trees'] for val in parse_trees]
