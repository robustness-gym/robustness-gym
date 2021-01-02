from typing import List

from robustnessgym import Identifier, SliceBuilder
from robustnessgym.core.constants import CURATION


class Curator(SliceBuilder):
    def __init__(
        self,
        identifiers: List[Identifier],
        apply_fn,
    ):
        super(Curator, self).__init__(
            category=CURATION,
            identifiers=identifiers,
            apply_fn=apply_fn,
        )

    def __call__(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        pass
