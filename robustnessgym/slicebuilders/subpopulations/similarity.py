from functools import partial
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.stats import spearmanr

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import strings_as_json
from robustnessgym.ops.similarity import RougeMatrix, RougeScore
from robustnessgym.slicebuilders.subpopulations.score import ScoreSubpopulation


class RougeScoreSubpopulation(ScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "fmeasure"),
        *args,
        **kwargs,
    ):
        super(RougeScoreSubpopulation, self).__init__(
            intervals=intervals,
            identifiers=[
                Identifier(
                    _name=self.__class__.__name__,
                    gte=interval[0],
                    lte=interval[1],
                    metric=metric,
                )
                for interval in intervals
            ],
            *args,
            **kwargs,
        )

        # Assign the metric
        self.metric = metric

    def score(
        self, batch: Dict[str, List], columns: List[str], *args, **kwargs
    ) -> np.ndarray:
        assert len(columns) == 2, "Must have exactly 2 columns."

        # Retrieve Rouge scores
        scores = RougeScore.retrieve(
            batch=batch,
            columns=columns,
            proc_fns=partial(RougeScore.select, metric=self.metric),
        )[strings_as_json(columns)]

        return np.array(scores)


class Abstractiveness(RougeScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "precision"),
    ):
        super(Abstractiveness, self).__init__(
            intervals=intervals,
            metric=metric,
        )


class Distillation(RougeScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "recall"),
    ):
        super(Distillation, self).__init__(
            intervals=intervals,
            metric=metric,
        )


class RougeMatrixScoreSubpopulation(ScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "fmeasure"),
        *args,
        **kwargs,
    ):
        assert (
            len(metric) == 2
        ), "Must pass in both rouge score and one of precision/recall/fmeasure."
        super(RougeMatrixScoreSubpopulation, self).__init__(
            intervals=intervals,
            identifiers=[
                Identifier(
                    _name=self.__class__.__name__,
                    gte=interval[0],
                    lte=interval[1],
                    metric=metric,
                )
                for interval in intervals
            ],
            *args,
            **kwargs,
        )

        # Assign the metric
        self.metric = metric

    def reduce(self, matrices: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def score(
        self, batch: Dict[str, List], columns: List[str], *args, **kwargs
    ) -> np.ndarray:
        assert len(columns) == 2, "Must have exactly 2 columns."

        # Retrieve the relevant Rouge matrices
        matrices = RougeMatrix.retrieve(
            batch=batch,
            columns=columns,
            proc_fns=partial(RougeMatrix.select, metric=self.metric),
        )[strings_as_json(columns)]

        return self.reduce(matrices)


class Position(RougeMatrixScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "fmeasure"),
    ):
        super(Position, self).__init__(
            intervals=intervals,
            metric=metric,
        )

    def reduce(self, matrices: List[np.ndarray]) -> np.ndarray:
        # Compute position of best-matched sentence in source document
        # Then compute mean position, capturing where position mostly comes from
        return np.array(
            [np.mean(np.argmax(mat, axis=0)) / mat.shape[0] for mat in matrices]
        )


class Dispersion(RougeMatrixScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "fmeasure"),
    ):
        super(Dispersion, self).__init__(
            intervals=intervals,
            metric=metric,
        )

    def reduce(self, matrices: List[np.ndarray]) -> np.ndarray:
        # Compute position of best-matched sentence in source document
        # Then compute std dev of position, capturing how spread out the positions are
        return np.array(
            [np.std(np.argmax(mat, axis=0) / mat.shape[0]) for mat in matrices]
        )


class Ordering(RougeMatrixScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "fmeasure"),
    ):
        super(Ordering, self).__init__(
            intervals=intervals,
            metric=metric,
        )

    def reduce(self, matrices: List[np.ndarray]) -> np.ndarray:
        # Compute position of best-matched sentence in source document
        # Then compute spearman correlation of position with range(..),
        # capturing whether the order of information is reversed
        return np.array(
            [
                spearmanr(
                    np.arange(mat.shape[1]) / mat.shape[0],
                    np.argmax(mat, axis=0) / mat.shape[0],
                )[0]
                for mat in matrices
            ]
        )


class NuclearNorm(RougeMatrixScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "fmeasure"),
    ):
        super(NuclearNorm, self).__init__(
            intervals=intervals,
            metric=metric,
        )

    def reduce(self, matrices: List[np.ndarray]) -> np.ndarray:
        return np.array(
            [
                np.sum(np.abs(svd(mat, full_matrices=False, compute_uv=False)))
                for mat in matrices
            ]
        )


class SpectralNorm(RougeMatrixScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "fmeasure"),
    ):
        super(SpectralNorm, self).__init__(
            intervals=intervals,
            metric=metric,
        )

    def reduce(self, matrices: List[np.ndarray]) -> np.ndarray:
        return np.array(
            [
                np.max(np.abs(svd(mat, full_matrices=False, compute_uv=False)))
                for mat in matrices
            ]
        )


class FrobeniusNorm(RougeMatrixScoreSubpopulation):
    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        metric: Sequence[str] = ("rouge1", "fmeasure"),
    ):
        super(FrobeniusNorm, self).__init__(
            intervals=intervals,
            metric=metric,
        )

    def reduce(self, matrices: List[np.ndarray]) -> np.ndarray:
        return np.array([norm(mat) for mat in matrices])
