import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Dict


def get_circle_coordinates(r: float, degree: float):
    if degree < 0 or degree > 360:
        raise ValueError

    radian = (degree / 360) * 2 * np.pi
    x = r * np.sin(radian)
    y = r * np.cos(radian)
    return x, y


def distance_to_points_on_circle(
        x: float, r: float,
        weights: List[float],
        points: List[List[float]]) -> float:
    # x^2 + y^2 = r^2
    y = np.sqrt(np.square(r) - np.square(x))

    weighted_distances = 0.0
    for weight, point in zip(weights, points):
        _x, _y = point
        distance = np.sqrt((np.square(x - _x) + np.square(y - _y)))
        weighted_distances += weight * distance

    return weighted_distances


def distance_to_points_within_circle(
        x_y: List[float],
        weights: List[float],
        points: List[List[float]]) -> float:
    if len(x_y) != 2:
        raise ValueError(f"Invalid `x_y` {x_y}")

    x, y = x_y
    weighted_distances = 0.0
    for weight, point in zip(weights, points):
        _x, _y = point
        distance = np.sqrt((np.square(x - _x) + np.square(y - _y)))
        weighted_distances += weight * distance

    return weighted_distances


def distance_to_points_within_circle_vectorized(
        x_y: List[float],
        weights: np.ndarray,
        points: np.ndarray) -> float:
    if len(x_y) != 2:
        raise ValueError(f"Invalid `x_y` {x_y}")
    if len(weights.shape) != 1:
        raise ValueError(f"Invalid `weights` shape {weights.shape}")
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid `points` shape {points.shape}")
    if weights.shape[0] != points.shape[0]:
        raise ValueError(f"Incompatible shapes {weights.shape} {points.shape}")

    point = np.array(x_y)
    distance = np.sqrt(np.square(point - points).sum(axis=-1))
    weighted_distances = distance * weights
    return weighted_distances.sum()


def get_within_circle_constraint(r: float) -> Callable[[List[float]], float]:

    # Inequality constraint must be non-negative
    def _constraint(x_y: List[float]) -> float:
        x, y = x_y
        return np.square(r) - np.square(x) - np.square(y)

    return _constraint


def plot_influences_distribution(
        influences_collections: List[Dict[int, float]],
        label: str) -> None:

    influences: List[float] = []
    for L in influences_collections:
        influences.extend(L.values())
    plt.hist(influences, label=label)
