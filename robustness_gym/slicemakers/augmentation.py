from typing import List

from robustness_gym.constants import AUGMENTATION
from robustness_gym.identifier import Identifier
from robustness_gym.slicemakers.transform import Transform


class Augmentation(Transform):

    def __init__(
            self,
            identifiers: List[Identifier],
            apply_fn=None,
    ):
        super(Augmentation, self).__init__(
            category=AUGMENTATION,
            identifiers=identifiers,
            apply_fn=apply_fn,
        )
