"""
Import common classes.
"""

from robustnessgym.cachedops.allen.allen_predictor import AllenPredictor
from robustnessgym.cachedops.allen.constituency_parser import AllenConstituencyParser
from robustnessgym.cachedops.allen.dependency_parser import AllenDependencyParser
from robustnessgym.cachedops.allen.semantic_role_labeler import AllenSemanticRoleLabeler
from robustnessgym.cachedops.bootleg import Bootleg
from robustnessgym.cachedops.similarity import (
    SentenceSimilarityMatrix,
    RougeMatrix,
    RougeScore,
)
from robustnessgym.slicebuilders.attacks.textattack import TextAttack
from robustnessgym.slicebuilders.transformations.fairseq import (
    FairseqBacktranslation,
)
from robustnessgym.slicebuilders.transformations.eda import (
    EasyDataAugmentation,
)
from robustnessgym.slicebuilders.transformations.nlpaug import (
    NlpAugTransformation,
)
from robustnessgym.slicebuilders.transformations.similarity import (
    RougeMatrixSentenceTransformation,
)
from robustnessgym.slicebuilders.slicebuilder import (
    SliceBuilder,
    SliceBuilderCollection,
)
from robustnessgym.slicebuilders.subpopulations.constituency_overlap import (
    ConstituencyOverlapSubpopulation,
    ConstituencySubtreeSubpopulation,
    FuzzyConstituencySubtreeSubpopulation,
)
from robustnessgym.slicebuilders.subpopulations.lexical_overlap import (
    LexicalOverlapSubpopulation,
)
from robustnessgym.slicebuilders.subpopulations.length import LengthSubpopulation
from robustnessgym.slicebuilders.subpopulations.entity_frequency import EntityFrequency
from robustnessgym.slicebuilders.subpopulations.hans import (
    HansAllPhrases,
    HansSingularNouns,
    HansPluralNouns,
    HansTransitiveVerbs,
    HansPassiveVerbs,
    HansIntransitiveVerbs,
    HansNPSVerbs,
    HansNPZVerbs,
    HansPluralNPZVerbs,
    HansPrepositions,
    HansConjs,
    HansPastParticiples,
    HansUnderstoodArgumentVerbs,
    HansNonEntQuotVerbs,
    HansQuestionEmbeddingVerbs,
    HansCalledObjects,
    HansToldObjects,
    HansFoodWords,
    HansLocationNounsA,
    HansLocationNounsB,
    HansWonObjects,
    HansReadWroteObjects,
    HansAdjectives,
    HansAdjectivesCompNonEnt,
    HansAdjectivesCompEnt,
    HansAdverbs,
    HansConstAdv,
    HansConstQuotEntailed,
    HansRelations,
    HansQuestions,
    HansNonEntComplementNouns,
    HansEntComplementNouns,
    HansAdvsNonEntailed,
    HansAdvsEntailed,
)
from robustnessgym.slicebuilders.subpopulations.phrase import (
    AhoCorasick,
    HasPhrase,
    HasAnyPhrase,
    HasAllPhrases,
    HasNegation,
    HasTemporalPreposition,
    HasComparison,
    HasQuantifier,
    HasDefiniteArticle,
    HasIndefiniteArticle,
    HasPosessivePreposition,
)
from robustnessgym.slicebuilders.subpopulations.similarity import (
    RougeScoreSubpopulation,
    RougeMatrixScoreSubpopulation,
    Abstractiveness,
    Distillation,
    Ordering,
    Dispersion,
    Position,
)
from robustnessgym.core.testbench import TestBench
from robustnessgym.core.cachedops import (
    CachedOperation,
    SingleColumnCachedOperation,
    stow,
)
from robustnessgym.cachedops.strip_text import StripText
from robustnessgym.cachedops.spacy import Spacy
from robustnessgym.cachedops.textblob import TextBlob
from robustnessgym.core.dataset import Dataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.slice import Slice
from .slicebuilders.attack import Attack
from .slicebuilders.curator import Curator
from .slicebuilders.subpopulation import (
    Subpopulation,
    SubpopulationCollection,
)
from robustnessgym.slicebuilders.subpopulations.score import (
    ScoreSubpopulation,
)
from robustnessgym.tasks.task import (
    Task,
    NaturalLanguageInference,
    BinaryNaturalLanguageInference,
    TernaryNaturalLanguageInference,
    Summarization,
    Sentiment,
    BinarySentiment,
    QuestionAnswering,
    ExtractiveQuestionAnswering,
)

# from .attacks import *
# from .augmentations import *
# from .cache import *
# from .cache import (
#     CachedOperation,
#     stow
# )
# from .dataset import Dataset
# from .identifier import Identifier
# from .model import Model
# from .report import Report
# from .slice import Slice
# from .slicebuilders import *
# from .slicebuilders.attacks.textattack.textattack import TextAttack
# from .slicebuilders.slicebuilder import (
#     SliceBuilder,
# )
# from .slicebuilders.subpopulations.constituency_overlap.constituency_overlap import (
#     HasConstituencyOverlap,
#     HasConstituencySubtree,
#     HasFuzzyConstituencySubtree,
# )
# from .slicebuilders.subpopulations.length.length import HasLength
# from .slicebuilders.subpopulations.ner.entity_frequency import EntityFrequency
# from .slicebuilders.subpopulations.phrase.hans import (
#     HansAllPhrases,
#     HansSingularNouns,
#     HansPluralNouns,
#     HansTransitiveVerbs,
#     HansPassiveVerbs,
#     HansIntransitiveVerbs,
#     HansNPSVerbs,
#     HansNPZVerbs,
#     HansPluralNPZVerbs,
#     HansPrepositions,
#     HansConjs,
#     HansPastParticiples,
#     HansUnderstoodArgumentVerbs,
#     HansNonEntQuotVerbs,
#     HansQuestionEmbeddingVerbs,
#     HansCalledObjects,
#     HansToldObjects,
#     HansFoodWords,
#     HansLocationNounsA,
#     HansLocationNounsB,
#     HansWonObjects,
#     HansReadWroteObjects,
#     HansAdjectives,
#     HansAdjectivesCompNonEnt,
#     HansAdjectivesCompEnt,
#     HansAdverbs,
#     HansConstAdv,
#     HansConstQuotEntailed,
#     HansRelations,
#     HansQuestions,
#     HansNonEntComplementNouns,
#     HansEntComplementNouns,
#     HansAdvsNonEntailed,
#     HansAdvsEntailed,
# )
# from .slicebuilders.subpopulations.phrase.phrase import (
#     AhoCorasick,
#     HasPhrase,
#     HasAnyPhrase,
#     HasAllPhrases,
# )
# from .slicebuilders.subpopulations.phrase.wordlists import (
#     HasCategoryPhrase
# )
# from .storage import PicklerMixin
# from .task import (
#     Task,
#     NaturalLanguageInference,
#     BinaryNaturalLanguageInference,
#     TernaryNaturalLanguageInference,
# )
# from .testbench.testbench import TestBench
# from .tools import (
#     recmerge,
#     persistent_hash,
# )
