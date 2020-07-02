from robustnessgym.tasks.task import QuestionAnswering, TernaryNaturalLanguageInference

dataset_to_task = {
    # Natural Language Inference
    # Ternary
    "snli": TernaryNaturalLanguageInference,
    # Question Answering
    "squad": QuestionAnswering,
}
