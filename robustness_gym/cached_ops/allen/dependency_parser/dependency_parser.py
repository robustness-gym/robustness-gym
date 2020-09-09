from typing import List

from robustness_gym.cached_ops.allen.allen_predictor import AllenPredictor


class AllenDependencyParser(AllenPredictor):

    def __init__(self,
                 device: str = None):
        super(AllenDependencyParser, self).__init__(
            path="https://storage.googleapis.com/allennlp-public-models/"
                 "biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
            device=device,
        )
    #
    # def apply(self, text_batch) -> List:
    #     # Apply the dependency parser
    #     parse_trees = self.predictor.predict_batch_json([
    #         {'sentence': text} for text in text_batch
    #     ])
    #
    #     # Extract the tree from the output of the dependency parser
    #     return [val['trees'] for val in parse_trees]
