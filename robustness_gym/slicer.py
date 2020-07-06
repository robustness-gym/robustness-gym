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
from robustness_gym.dataset import Dataset
from robustness_gym.slice import Slice


class Slicer(ABC):

    def __init__(self):
        super().__init__()

        self.type = None
        self.tasks = None
        self.metadata = {}

    def slice_dataset(self, dataset: Dataset) -> List[Slice]:
        pass

    def slice_batch(self, batch: Dict[List]) -> List[Dict[List]]:
        pass


class Augmentation(Slicer):

    def __init__(self):
        super(Augmentation, self).__init__()

    def slice_dataset(self, dataset: Dataset) -> List[Slice]:
        pass


class AdversarialAttack(Slicer):

    def __init__(self):
        super(AdversarialAttack, self).__init__()

    def slice_dataset(self, dataset: Dataset) -> List[Slice]:
        pass


class Filter(Slicer):

    def __init__(self):
        super(Filter, self).__init__()

    def slice_dataset(self, dataset: Dataset) -> List[Slice]:
        pass
