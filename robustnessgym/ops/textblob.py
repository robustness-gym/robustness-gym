from __future__ import annotations

import subprocess
from typing import Collection, List

from meerkat import AbstractCell
from meerkat.tools.lazy_loader import LazyLoader

from robustnessgym.core.operation import Operation
from robustnessgym.core.slice import SliceDataPanel as DataPanel

textblob = LazyLoader(
    "textblob",
    error="TextBlob not available for import. Install using "
    "\npip install textblob\npython -m textblob.download_corpora",
)


class LazyTextBlobCell(AbstractCell):
    def __init__(self, text: str):
        super(LazyTextBlobCell, self).__init__()
        self.text = text

    def get(self, *args, **kwargs):
        return textblob.TextBlob(self.text)

    @property
    def data(self) -> object:
        return self.text

    @classmethod
    def _state_keys(cls) -> Collection:
        return {"text"}

    def __repr__(self):
        snippet = f"{self.text[:15]}..." if len(self.text) > 20 else self.text
        return f"{self.__class__.__name__}({snippet})"


class LazyTextBlobOp(Operation):
    def __init__(self):
        self._download()
        super(LazyTextBlobOp, self).__init__()

    @staticmethod
    def _download():
        subprocess.call(["python", "-m", "textblob.download_corpora"])

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> tuple:
        return ([LazyTextBlobCell(text) for text in dp[columns[0]]],)
