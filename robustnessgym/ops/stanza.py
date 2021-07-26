from __future__ import annotations

from typing import Dict, List

from meerkat import AbstractCell
from meerkat.tools.lazy_loader import LazyLoader

from robustnessgym.core.operation import Operation
from robustnessgym.core.slice import SliceDataPanel as DataPanel

stanza = LazyLoader("stanza", error="Please `pip install stanza`.")


class StanzaCell(AbstractCell):
    """Cell that stores a Stanza Document."""

    def __init__(self, doc: stanza.Document, *args, **kwargs):
        super(StanzaCell, self).__init__(*args, **kwargs)
        self.doc = doc

    def get(self, *args, **kwargs):
        return self.doc

    @property
    def data(self) -> stanza.Document:
        return self.doc

    @classmethod
    def from_text(cls, text: str, pipeline: stanza.Pipeline) -> StanzaCell:
        return cls(pipeline(text))

    def get_state(self) -> Dict:
        return {"doc": self.doc.to_serialized()}

    @classmethod
    def from_state(cls, state, **kwargs) -> StanzaCell:
        return cls(stanza.Document.from_serialized(state["doc"]))

    def __repr__(self):
        snippet = (
            f"{self.doc.text[:15]}..." if len(self.doc.text) > 20 else self.doc.text
        )
        return f"{self.__class__.__name__}({snippet})"

    def _repr_html_(self):
        return self.__repr__()

    def __getattr__(self, item):
        try:
            if item in {
                "lemma",
                "text",
                "upos",
                "xpos",
                "feats",
                "head",
                "deprel",
                "feats",
                "misc",
                "id",
            }:
                return self.doc.get(item)
        except AttributeError:
            return super().__getattr__(item)


class StanzaOp(Operation):
    """Operation that runs the Stanza pipeline.

    Stanza: https://stanfordnlp.github.io/stanza/
    """

    def __init__(self):
        super(StanzaOp, self).__init__()

        self._download()
        self.nlp = stanza.Pipeline()

    @staticmethod
    def _download():
        stanza.download()

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> tuple:
        """Process text examples by running them through the Stanza pipeline.

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            **kwargs: optional keyword arguments

        Returns:
            Tuple with single output, a list of StanzaCell objects.
        """

        return tuple(
            [
                [StanzaCell.from_text(text, self.nlp) for text in dp[col]]
                for col in columns
            ]
        )


# def input_columns(self):
#     return ['image', 'bb']

# op(dp, image_column=['image'], [{'image': '])

# op = StanzaOp()
# op(dp, 'passage') --> 1 output column
# op(dp, ['passage', 'question']) --> 2 output columns
