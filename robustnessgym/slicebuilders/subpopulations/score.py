from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from robustnessgym.core.cachedops import ScoreOperation
from robustnessgym.core.dataset import Batch, Dataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.slicebuilders.subpopulation import Subpopulation


class BinningMixin:
    def __init__(
        self,
        intervals: List[Tuple[Union[int, float, str], Union[int, float, str]]],
        bin_creation_fn: Callable = None,
        bin_fn: Callable = None,
        *args,
        **kwargs
    ):
        super(BinningMixin, self).__init__(*args, **kwargs)

        # Set the intervals
        self.intervals = intervals
        self.left_limits = None
        self.right_limits = None

        # Keep track of scores
        self.scores = []

        # Assign the bin fns
        if bin_creation_fn and bin_fn:
            self.create_bins = bin_creation_fn
            self.bin = bin_fn

    def _reset_scores(self):
        self.scores = []

    def replace_percentile(self, limit):
        if isinstance(limit, str) and limit.endswith("%"):
            return np.percentile(self.scores, float(limit.replace("%", "")))
        elif isinstance(limit, float) or isinstance(limit, int):
            return limit
        else:
            raise NotImplementedError

    def create_bins(self):
        for i in range(len(self.intervals)):
            (left_limit, right_limit) = self.intervals[i]
            self.intervals[i] = (
                self.replace_percentile(left_limit),
                self.replace_percentile(right_limit),
            )

        self.left_limits = np.array([interval[0] for interval in self.intervals])
        self.right_limits = np.array([interval[1] for interval in self.intervals])

    def bin(self, scores: List[Union[int, float]]) -> np.ndarray:
        # Convert to np.ndarry
        scores = np.array(scores)

        # Bin the scores
        return (
            (self.left_limits <= scores[:, np.newaxis])
            & (scores[:, np.newaxis] <= self.right_limits)
        ).astype(np.int32)


class ScoreSubpopulation(Subpopulation, BinningMixin):
    def __init__(
        self,
        intervals: List[Tuple[Union[int, float, str], Union[int, float, str]]],
        identifiers: List[Identifier] = None,
        score_fn: Callable = None,
        bin_creation_fn: Callable = None,
        bin_fn: Callable = None,
        *args,
        **kwargs
    ):

        if not identifiers:
            if score_fn:
                identifiers = [
                    Identifier(
                        _name=self.__class__.__name__,
                        gte=interval[0],
                        lte=interval[1],
                        score_fn=score_fn,
                    )
                    for interval in intervals
                ]
            else:
                identifiers = [
                    Identifier(
                        _name=self.__class__.__name__, gte=interval[0], lte=interval[1]
                    )
                    for interval in intervals
                ]

        super(ScoreSubpopulation, self).__init__(
            intervals=intervals,
            identifiers=identifiers,
            bin_creation_fn=bin_creation_fn,
            bin_fn=bin_fn,
            *args,
            **kwargs,
        )

        # Assign the score fn
        if score_fn:
            self.score = score_fn

    def prepare_dataset(
        self,
        dataset: Dataset,
        columns: List[str],
        batch_size: int = 32,
        *args,
        **kwargs
    ) -> None:

        # First reset the scores
        self._reset_scores()

        # Prepare the dataset
        super(ScoreSubpopulation, self).prepare_dataset(
            dataset=dataset,
            columns=columns,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

        # Create the bins
        self.create_bins()

    def prepare_batch(self, batch: Batch, columns: List[str], *args, **kwargs) -> None:

        # Compute the scores
        if isinstance(self.score, ScoreOperation):
            self.scores.extend(self.score.retrieve(batch=batch, columns=columns))
        elif isinstance(self.score, Callable):
            self.scores.extend(self.score(batch=batch, columns=columns))
        else:
            raise RuntimeError("score function invalid.")

    def score(
        self, batch: Dict[str, List], columns: List[str], *args, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError("Return a vector of float scores for each example.")

    def apply(
        self,
        slice_membership: np.ndarray,
        batch: Dict[str, List],
        columns: List[str],
        *args,
        **kwargs
    ) -> np.ndarray:
        # Keep track of the score of each example
        if isinstance(self.score, ScoreOperation):
            scores = self.score.retrieve(batch=batch, columns=columns)
        elif isinstance(self.score, Callable):
            scores = self.score(batch=batch, columns=columns)
        else:
            raise RuntimeError("score function invalid.")

        assert (
            len(scores) == slice_membership.shape[0]
        ), "Must have exactly one score per example."

        return self.bin(scores=scores)


class MultiScoreSubpopulation(Subpopulation, BinningMixin):
    def __init__(
        self,
        intervals: List[Tuple[Union[int, float, str], Union[int, float, str]]],
        identifiers: List[Identifier] = None,
        score_fn: Callable = None,
        bin_creation_fn: Callable = None,
        bin_fn: Callable = None,
        *args,
        **kwargs
    ):

        if not identifiers:
            if score_fn:
                identifiers = [
                    Identifier(
                        _name=self.__class__.__name__,
                        gte=interval[0],
                        lte=interval[1],
                        score_fn=score_fn,
                    )
                    for interval in intervals
                ]
            else:
                identifiers = [
                    Identifier(
                        _name=self.__class__.__name__, gte=interval[0], lte=interval[1]
                    )
                    for interval in intervals
                ]

        super(MultiScoreSubpopulation, self).__init__(
            intervals=intervals,
            identifiers=identifiers,
            bin_creation_fn=bin_creation_fn,
            bin_fn=bin_fn,
            *args,
            **kwargs,
        )

        # Assign the score fn
        if score_fn:
            self.score = score_fn

    def prepare_dataset(
        self,
        dataset: Dataset,
        columns: List[str],
        batch_size: int = 32,
        *args,
        **kwargs
    ) -> None:

        # First reset the scores
        self._reset_scores()

        # Prepare the dataset
        super(MultiScoreSubpopulation, self).prepare_dataset(
            dataset=dataset,
            columns=columns,
            batch_size=batch_size,
        )

        # Create the bins
        self.create_bins()

    def prepare_batch(self, batch: Batch, columns: List[str], *args, **kwargs) -> None:

        # Compute the scores
        if isinstance(self.score, ScoreOperation):
            self.scores.extend(self.score.retrieve(batch=batch, columns=columns))
        elif isinstance(self.score, Callable):
            self.scores.extend(self.score(batch=batch, columns=columns))
        else:
            raise RuntimeError("score function invalid.")
