from robustnessgym.cachedops.allen.allen_predictor import AllenPredictor


class AllenSemanticRoleLabeler(AllenPredictor):
    def __init__(self, device: str = None):
        super(AllenSemanticRoleLabeler, self).__init__(
            path="https://storage.googleapis.com/allennlp-public-models/bert-base-srl"
            "-2020.03.24.tar.gz",
            device=device,
        )
