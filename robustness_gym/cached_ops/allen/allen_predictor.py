from typing import List

import torch
from allennlp.predictors import Predictor

from robustness_gym.cached_ops.cached_ops import CachedOperation


class AllenPredictor(CachedOperation):

    def __init__(self,
                 path: str,
                 device: str):
        super(AllenPredictor, self).__init__()

        # If no device is passed in, automatically use GPU if available
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Resolve the device
        cuda_device = -1
        if device.startswith('cuda'):
            cuda_device = 0 if ":" not in device else int(device.split(":")[-1])

        # Set up Allen's predictor
        self._predictor = Predictor.from_path(
            archive_path=path,
            cuda_device=cuda_device
        )

    @property
    def predictor(self):
        return self._predictor

    def apply(self, text_batch) -> List:
        # Apply the semantic role labeler
        return self.predictor.predict_batch_json([
            {'sentence': text} for text in text_batch
        ])
