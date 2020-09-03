from typing import List

from robustness_gym import CachedOperation


class TextBlob(CachedOperation):

    def __init__(self):
        super(TextBlob, self).__init__(identifier='TextBlob')

    def apply(self, text_batch) -> List:
        pass