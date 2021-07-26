from functools import partial
from typing import List, Sequence, Tuple

import numpy as np

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.core.tools import strings_as_json
from robustnessgym.ops.similarity import RougeMatrix
from robustnessgym.ops.spacy import SpacyOp
from robustnessgym.slicebuilders.transformation import Transformation


class RougeMatrixSentenceTransformation(Transformation):
    def __init__(self, metric: Sequence[str], threshold: float):
        super(RougeMatrixSentenceTransformation, self).__init__(
            num_transformed=1,
            identifiers=Identifier.range(n=1, _name=self.__class__.__name__),
        )

        self.metric = metric
        self.threshold = threshold

    def apply(
        self,
        batch: DataPanel,
        columns: List[str],
        skeleton_batches: List[DataPanel],
        slice_membership: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[List[DataPanel], np.ndarray]:
        assert len(columns) == 2

        # Retrieve the relevant Rouge matrices
        matrices = RougeMatrix.retrieve(
            batch=batch,
            columns=columns,
            proc_fns=partial(RougeMatrix.select, metric=self.metric),
        )[strings_as_json(columns)]

        # Find max value along each row, remove rows that have max value below a
        # threshold
        rows_to_keep = [
            (m / (m.sum(axis=0) + 1e-5)).max(axis=1) >= self.threshold for m in matrices
        ]

        # Fetch sentences for the first column
        sentences = SpacyOp.retrieve(
            batch=batch,
            columns=[columns[0]],
            proc_fns=SpacyOp.sentences,
        )[columns[0]]

        # Delete sentences
        new_sentences = [
            " ".join(np.array(sent)[rows_to_keep[i]])
            for i, sent in enumerate(sentences)
        ]

        # Store the augmented text in the skeleton batches
        for i, augmented in enumerate(new_sentences):
            skeleton_batches[0][columns[0]][i] = augmented

        return skeleton_batches, slice_membership
