from robustnessgym.cachedops.allen.allen_predictor import AllenPredictor


class AllenConstituencyParser(AllenPredictor):
    def __init__(self, device: str = None, *args, **kwargs):
        super(AllenConstituencyParser, self).__init__(
            path="https://storage.googleapis.com/allennlp-public-models/elmo"
            "-constituency-parser-2020.02.10.tar.gz",
            device=device,
            *args,
            **kwargs,
        )

    @classmethod
    def encode(cls, prediction) -> str:
        # Extract the tree from the output of the constituency parser
        return super().encode(obj=prediction["trees"])
