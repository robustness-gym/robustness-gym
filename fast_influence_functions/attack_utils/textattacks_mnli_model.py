import sys
sys.path.insert(0, "/workspace/hguo-scratchpad")

from experiments.mnli_utils import create_tokenizer_and_model

tokenizer, model = create_tokenizer_and_model(
    "/export/home/Experiments/20200706/")
