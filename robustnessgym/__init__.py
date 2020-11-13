"""
Import common classes.
"""

from robustnessgym.cached_ops.allen.allen_predictor import (
    AllenPredictor
)
from robustnessgym.cached_ops.allen.constituency_parser.constituency_parser import (
    AllenConstituencyParser
)
from robustnessgym.cached_ops.allen.dependency_parser.dependency_parser import (
    AllenDependencyParser
)
from robustnessgym.cached_ops.allen.semantic_role_labeler.semantic_role_labeler import (
    AllenSemanticRoleLabeler
)
from robustnessgym.cached_ops.bootleg.bootleg import (
    Bootleg
)
from robustnessgym.cached_ops.similarity.similarity import (
    SentenceSimilarityMatrix,
    RougeMatrix,
)
from robustnessgym.slicebuilders.attacks.textattack.textattack import TextAttack
from robustnessgym.slicebuilders.augmentations.backtranslation.fairseq import (
    FairseqBacktranslation,
)
from robustnessgym.slicebuilders.augmentations.eda.eda import (
    EasyDataAugmentation,
)
from robustnessgym.slicebuilders.augmentations.nlpaug.nlpaug import (
    NlpAug,
)
from robustnessgym.slicebuilders.augmentations.similarity.similarity import (
    RougeMatrixSentenceAugmentation,
)
from robustnessgym.slicebuilders.slicebuilder import (
    SliceBuilder,
    SliceBuilderCollection,
)
from robustnessgym.slicebuilders.subpopulations.constituency_overlap.constituency_overlap import (
    ConstituencyOverlapSubpopulation,
    ConstituencySubtreeSubpopulation,
    FuzzyConstituencySubtreeSubpopulation,
)
from robustnessgym.slicebuilders.subpopulations.lexical_overlap.lexical_overlap import (
    LexicalOverlapSubpopulation,
)
from robustnessgym.slicebuilders.subpopulations.length.length import LengthSubpopulation
from robustnessgym.slicebuilders.subpopulations.ner.entity_frequency import EntityFrequency
from robustnessgym.slicebuilders.subpopulations.phrase.hans import (
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
from robustnessgym.slicebuilders.subpopulations.phrase.phrase import (
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
from robustnessgym.slicebuilders.subpopulations.phrase.wordlists import (
    HasCategoryPhrase
)
from robustnessgym.slicebuilders.subpopulations.similarity.similarity import (
    RougeScoreSubpopulation,
    RougeMatrixScoreSubpopulation,
)
from robustnessgym.testbench.testbench import TestBench
from .cached_ops.cached_ops import (
    CachedOperation,
    SingleColumnCachedOperation,
    stow
)
from .cached_ops.custom.strip_text import StripText
from .cached_ops.spacy.spacy import Spacy
from .cached_ops.textblob.textblob import TextBlob
from .dataset import Dataset
from .identifier import Identifier
from .model import Model
from .report import Report
from .slice import Slice
from .slicebuilders.attack import Attack
from .slicebuilders.augmentation import Augmentation
from .slicebuilders.curator import Curator
from .slicebuilders.subpopulation import (
    Subpopulation,
    SubpopulationCollection,
)
from .slicebuilders.subpopulations.score.score import (
    ScoreSubpopulation,
)
from .storage import StorageMixin
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
from .tools import (
    recmerge,
    persistent_hash,
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
