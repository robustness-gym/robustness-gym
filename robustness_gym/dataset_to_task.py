from robustness_gym.task import *

dataset_to_task = {
    # Natural Language Inference

    # Ternary
    'snli': TernaryNaturalLanguageInference,

    # Question Answering
    'squad': QuestionAnswering,

}
