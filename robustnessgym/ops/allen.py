from typing import List

from meerkat.tools.lazy_loader import LazyLoader

from robustnessgym.core.operation import Operation
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.mixins.device import DeviceMixin

predictors = LazyLoader("allennlp.predictors")


class AllenPredictionOp(DeviceMixin, Operation):
    def __init__(
        self,
        path: str,
        device: str,
    ):
        super(AllenPredictionOp, self).__init__(device=device)

        self._predictor = predictors.Predictor.from_path(
            archive_path=path, cuda_device=self.cuda_device
        )

    @property
    def predictor(self):
        return self._predictor

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> tuple:
        return (
            self.predictor.predict_batch_json(
                [{"sentence": text} for text in dp[columns[0]]]
            ),
        )


class AllenConstituencyParsingOp(AllenPredictionOp):
    def __init__(self, device: str = None):
        super(AllenConstituencyParsingOp, self).__init__(
            path="https://storage.googleapis.com/allennlp-public-models/elmo"
            "-constituency-parser-2020.02.10.tar.gz",
            device=device,
        )

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> tuple:
        return (
            [
                p["trees"]
                for p in self.predictor.predict_batch_json(
                    [{"sentence": text} for text in dp[columns[0]]]
                )
            ],
        )


class AllenDependencyParsingOp(AllenPredictionOp):
    def __init__(self, device: str = None):
        super(AllenDependencyParsingOp, self).__init__(
            path="https://storage.googleapis.com/allennlp-public-models/"
            "biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
            device=device,
        )


class AllenSemanticRoleLabelingOp(AllenPredictionOp):
    def __init__(self, device: str = None):
        super(AllenSemanticRoleLabelingOp, self).__init__(
            path="https://storage.googleapis.com/allennlp-public-models/bert-base-srl"
            "-2020.03.24.tar.gz",
            device=device,
        )
