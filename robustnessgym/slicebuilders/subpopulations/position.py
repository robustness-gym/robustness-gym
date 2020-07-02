# from typing import Callable, Dict, List, Tuple
#
# import numpy as np
#
# from robustnessgym.core.identifier import Identifier
# from robustnessgym.slicebuilders.subpopulation import Subpopulation
#
#
# class Position:
#     def __init__(
#         self,
#         identifiers: List[Identifier] = None,
#         position_fn: Callable = None,
#         *args,
#         **kwargs
#     ):
#
#         if not identifiers:
#             identifiers = [
#                 Identifier(
#                     _name=self.__class__.__name__,
#                     gte=interval[0],
#                     lte=interval[1],
#                     score_fn=score_fn,
#                 )
#                 for interval in intervals
#             ]
#
#         # Set the intervals
#         self.intervals = intervals
#         self.left_limits = np.array([interval[0] for interval in intervals])
#         self.right_limits = np.array([interval[1] for interval in intervals])
#
#         # Assign the score fn
#         if score_fn:
#             self.score = score_fn
#
#     def score(
#         self, batch: Dict[str, List], keys: List[str], *args, **kwargs
#     ) -> np.ndarray:
#         raise NotImplementedError("Return a vector of float scores for each example.")
#
#     def apply(
#         self,
#         slice_membership: np.ndarray,
#         batch: Dict[str, List],
#         keys: List[str],
#         *args,
#         **kwargs
#     ) -> np.ndarray:
#         # Keep track of the score of each example
#         scores = self.score(batch=batch, keys=keys)
#
#         assert scores.shape == (
#             slice_membership.shape[0],
#         ), "Must have exactly one score per example."
#
#         return (
#             (self.left_limits <= scores[:, np.newaxis])
#             & (scores[:, np.newaxis] <= self.right_limits)
#         ).astype(np.int32)
