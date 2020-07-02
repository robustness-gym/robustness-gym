from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from robustnessgym.slicebuilders.transformation import Transformation


class GPT3Transformation(Transformation):
    def __init__(
        self,
    ):
        super(GPT3Transformation, self).__init__()

    def apply(
        self,
        skeleton_batches: List[Dict[str, List]],
        slice_membership: np.ndarray,
        batch: Dict[str, List],
        columns: List[str],
        *args,
        **kwargs
    ) -> Tuple[List[Dict[str, List]], np.ndarray]:
        pass
