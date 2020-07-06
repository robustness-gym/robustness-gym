from __future__ import annotations

from abc import ABCMeta, ABC
from functools import partial
from typing import *

import spacy
import cytoolz as tz
from pyarrow import json
import nlp
import streamlit as st
import torch
from allennlp.predictors import Predictor
import allennlp_models.structured_prediction.predictors.constituency_parser
from quinine.common.utils import rmerge

from robustness_gym.dataset import *


class Slice(Dataset):

    # def __init__(self):
    #     # A slice contains information about how it was derived
    #     self.info = {
    #
    #         'type': [],  # ['augmentation', 'adv_attack', 'eval_set', 'sfs', 'dataset'],
    #         'tasks': [],  # ['NLI'],
    #
    #         'split': None,  # ['train', 'val', 'test'] --> val is most likely for our library
    #
    #         # Dependencies is
    #         'dependencies': {},  # dependence find a better word
    #
    #     }

    def add_examples(self, examples: Sequence[Example]) -> None:
        self.examples.append(examples)


