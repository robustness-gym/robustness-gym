from typing import List

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.constants import *
from robustnessgym.slicebuilders.transform import Transform


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
