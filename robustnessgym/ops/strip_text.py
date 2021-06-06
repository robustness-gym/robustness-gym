from typing import List

from robustnessgym.core.operation import Operation
from robustnessgym.core.slice import SliceDataPanel as DataPanel


class StripTextOp(Operation):
    def __init__(self):
        super(StripTextOp, self).__init__()

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> tuple:
        # Clean up each text with a simple function and return the stripped text
        return (
            list(
                map(
                    lambda text: text.lower()
                    .replace(".", "")
                    .replace("?", "")
                    .replace("!", "")
                    .replace(",", ""),
                    dp[columns[0]],
                )
            ),
        )
