"""
Import common classes.
"""

from robustness_gym.cached_ops.allen.constituency_parser.constituency_parser import (
    AllenConstituencyParser
)
from robustness_gym.slicemakers.augmentations.eda.eda import (
    EasyDataAugmentation,
)
from robustness_gym.slicemakers.slicemaker import (
    SliceMaker,
)
from robustness_gym.slicemakers.subpopulations.phrase.phrase import (
    HasPhrase,
    HasAnyPhrase,
    HasAllPhrases,
)
from robustness_gym.slicemakers.attacks.textattack.textattack import TextAttack
from robustness_gym.testbench.testbench import TestBench
from .cached_ops.cached_ops import (
    CachedOperation,
    stow
)
from .cached_ops.custom.strip_text import StripText
from .cached_ops.spacy.spacy import Spacy
from .dataset import Dataset
from .identifier import Identifier
from .model import Model
from .report import Report
from .slice import Slice
from .slicemakers.attack import Attack
from .slicemakers.augmentation import Augmentation
from .slicemakers.curator import Curator
from .slicemakers.subpopulation import Subpopulation
from .storage import PicklerMixin
from .task import (
    Task,
    NaturalLanguageInference,
    BinaryNaturalLanguageInference,
    TernaryNaturalLanguageInference,
)
from .tools import (
    recmerge,
    persistent_hash,
)
