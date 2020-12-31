from typing import List

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.constants import *
from robustnessgym.slicebuilders.transformation import Transformation


class Attack(Transformation):

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
