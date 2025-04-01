"""
Simplification algorithms for polygon regularization.

Contains the Ramer-Douglas-Peucker algorithm implementation for line simplification.
"""

from typing import Callable

import numpy as np


def point_to_line_distance(
    point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
) -> float:
    """
    Calculates the perpendicular distance from a point to a line defined by two points.

    Parameters:
    -----------
    point : array-like
        The point to calculate distance from
    line_start : array-like
        Starting point of the line
    line_end : array-like
        Ending point of the line

    Returns:
    --------
    float
        Distance from point to line
    """
    point, line_start, line_end = point[:2], line_start[:2], line_end[:2]  # ensure 2D
    if line_start[0] == line_end[0]:
        return np.abs(point[0] - line_start[0])

    return np.divide(
        np.linalg.norm(np.linalg.det([line_end - line_start, line_start - point])),
        np.linalg.norm(line_end - line_start),
    )


def _ramer_douglas_peucker_recursive(
    points: np.ndarray, epsilon: float, distance_func: Callable
) -> np.ndarray:
    """
    Recursive implementation of the Ramer-Douglas-Peucker algorithm.

    Parameters:
    -----------
    points : numpy.ndarray
        Array of points to simplify
    epsilon : float
        Simplification threshold
    distance_func : function
        Function to calculate point-to-line distance

    Returns:
    --------
    numpy.ndarray
        Simplified points
    """
    max_distance = 0.0
    index_of_furthest = -1

    # Find point with max distance from line between first and last point
    for i in range(1, points.shape[0]):
        distance = distance_func(points[i], points[0], points[-1])
        if distance > max_distance:
            index_of_furthest = i
            max_distance = distance

    # If max distance exceeds epsilon, recursively simplify
    if max_distance > epsilon:
        # Recursive call for first segment
        first_segment = _ramer_douglas_peucker_recursive(
            points[: index_of_furthest + 1], epsilon, distance_func
        )
        # Recursive call for second segment
        second_segment = _ramer_douglas_peucker_recursive(
            points[index_of_furthest:], epsilon, distance_func
        )
        # Combine simplified segments
        return np.vstack((first_segment[:-1], second_segment))
    else:
        # If max distance is below epsilon, just use endpoints
        return np.vstack((points[0], points[-1]))
