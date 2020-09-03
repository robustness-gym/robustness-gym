from typing import List

from robustness_gym.identifier import Identifier
from robustness_gym.constants import *
from robustness_gym.slicemakers.transform import Transform


class Attack(Transform):

    def __init__(
            self,
            identifiers: List[Identifier],
            apply_fn=None,
    ):
        super(Attack, self).__init__(
            category=ATTACK,
            identifiers=identifiers,
            apply_fn=apply_fn,
        )
