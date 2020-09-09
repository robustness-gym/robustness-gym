from typing import List

from robustness_gym import CachedOperation


class StripText(CachedOperation):

    def __init__(self):
        super(StripText, self).__init__()

    def apply(self, text_batch) -> List:
        # Clean up each text with a simple function
        stripped = list(
            map(
                lambda text: text.lower().replace(".", "").replace("?", "").replace("!", "").replace(",", ""),
                text_batch
            )
        )

        # Return the stripped sentences
        return stripped
