"""Import common classes."""
# flake8: noqa
from meerkat.provenance import set_provenance

from robustnessgym.logging.utils import (
    initialize_logging,
    set_logging_level,
    set_logging_level_for_imports,
)

initialize_logging()
set_provenance()


from robustnessgym.core.devbench import DevBench
from robustnessgym.core.identifier import Id, Identifier
from robustnessgym.core.model import HuggingfaceModel, LudwigModel, Model
from robustnessgym.core.operation import Operation, lookup
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.core.testbench import TestBench

# from robustnessgym.ops.allen import (
#     AllenConstituencyParsingOp,
#     AllenDependencyParsingOp,
#     AllenPredictionOp,
#     AllenSemanticRoleLabelingOp,
# )
# from robustnessgym.ops.bootleg import BootlegAnnotatorOp
# from robustnessgym.ops.similarity import (
#     RougeMatrix,
#     RougeScore,
#     SentenceSimilarityMatrixOp,
# )
# from robustnessgym.ops.spacy import SpacyOp
# from robustnessgym.ops.stanza import StanzaOp
# from robustnessgym.ops.strip_text import StripTextOp
# from robustnessgym.ops.textblob import LazyTextBlobOp
from robustnessgym.slicebuilders.attacks.textattack import TextAttack
from robustnessgym.slicebuilders.slicebuilder import (  # SliceBuilderCollection,
    SliceBuilder,
)
from robustnessgym.slicebuilders.subpopulations.constituency_overlap import (
    ConstituencyOverlapSubpopulation,
    ConstituencySubtreeSubpopulation,
    FuzzyConstituencySubtreeSubpopulation,
)

# from robustnessgym.slicebuilders.subpopulations.entity_frequency
# import EntityFrequency
from robustnessgym.slicebuilders.subpopulations.hans import (
    HansAdjectives,
    HansAdjectivesCompEnt,
    HansAdjectivesCompNonEnt,
    HansAdverbs,
    HansAdvsEntailed,
    HansAdvsNonEntailed,
    HansAllPhrases,
    HansCalledObjects,
    HansConjs,
    HansConstAdv,
    HansConstQuotEntailed,
    HansEntComplementNouns,
    HansFoodWords,
    HansIntransitiveVerbs,
    HansLocationNounsA,
    HansLocationNounsB,
    HansNonEntComplementNouns,
    HansNonEntQuotVerbs,
    HansNPSVerbs,
    HansNPZVerbs,
    HansPassiveVerbs,
    HansPastParticiples,
    HansPluralNouns,
    HansPluralNPZVerbs,
    HansPrepositions,
    HansQuestionEmbeddingVerbs,
    HansQuestions,
    HansReadWroteObjects,
    HansRelations,
    HansSingularNouns,
    HansToldObjects,
    HansTransitiveVerbs,
    HansUnderstoodArgumentVerbs,
    HansWonObjects,
)
from robustnessgym.slicebuilders.subpopulations.length import NumTokensSubpopulation
from robustnessgym.slicebuilders.subpopulations.lexical_overlap import (
    LexicalOverlapSubpopulation,
)
from robustnessgym.slicebuilders.subpopulations.phrase import (
    AhoCorasick,
    HasAllPhrases,
    HasAnyPhrase,
    HasComparison,
    HasDefiniteArticle,
    HasIndefiniteArticle,
    HasNegation,
    HasPhrase,
    HasPosessivePreposition,
    HasQuantifier,
    HasTemporalPreposition,
)
from robustnessgym.slicebuilders.subpopulations.score import (
    BinarySubpopulation,
    IntervalSubpopulation,
    PercentileSubpopulation,
    ScoreSubpopulation,
)
from robustnessgym.slicebuilders.subpopulations.similarity import (
    Abstractiveness,
    Dispersion,
    Distillation,
    Ordering,
    Position,
    RougeMatrixScoreSubpopulation,
    RougeScoreSubpopulation,
)
from robustnessgym.slicebuilders.transformations.eda import EasyDataAugmentation
from robustnessgym.slicebuilders.transformations.fairseq import FairseqBacktranslation
from robustnessgym.slicebuilders.transformations.nlpaug import NlpAugTransformation
from robustnessgym.slicebuilders.transformations.similarity import (
    RougeMatrixSentenceTransformation,
)
from robustnessgym.tasks.task import (
    BinaryNaturalLanguageInference,
    BinarySentiment,
    ExtractiveQuestionAnswering,
    NaturalLanguageInference,
    QuestionAnswering,
    SentimentClassification,
    Summarization,
    Task,
    TernaryNaturalLanguageInference,
)

from .slicebuilders.attack import Attack
