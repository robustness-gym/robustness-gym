from typing import List

from robustnessgym.core.cachedops import SingleColumnCachedOperation


class StripText(SingleColumnCachedOperation):
    def __init__(self):
        super(StripText, self).__init__()

    def single_column_apply(self, column_batch: List, *args, **kwargs) -> List:
        # Clean up each text with a simple function and return the stripped text
        return list(
            map(
                lambda text: text.lower()
                .replace(".", "")
                .replace("?", "")
                .replace("!", "")
                .replace(",", ""),
                column_batch,
            )
        )
