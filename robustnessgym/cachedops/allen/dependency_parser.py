from robustnessgym.cachedops.allen.allen_predictor import AllenPredictor


class AllenDependencyParser(AllenPredictor):
    def __init__(self, device: str = None):
        super(AllenDependencyParser, self).__init__(
            path="https://storage.googleapis.com/allennlp-public-models/"
            "biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
            device=device,
        )
