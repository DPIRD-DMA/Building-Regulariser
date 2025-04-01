"""
Rotation functions for geometric operations.

Functions for rotating points clockwise and counterclockwise around a center point.
"""

import math
from typing import Tuple, Union

import numpy as np


def rotate_point_clockwise(
    point: Union[np.ndarray, Tuple[float, float]],
    center: Union[np.ndarray, Tuple[float, float]],
    angle_degrees: float,
) -> Tuple[float, float]:
    """
    Rotate a point clockwise around a center point

    Parameters:
    -----------
    point : array-like
        Point to rotate
    center : array-like
        Center of rotation
    angle_degrees : float
        Rotation angle in degrees

    Returns:
    --------
    tuple
        Rotated point coordinates
    """
    x, y = point
    center_x, center_y = center
    angle_radians = math.radians(angle_degrees)

    # Translate point to origin
    translated_x = x - center_x
    translated_y = y - center_y

    # Rotate
    rotated_x = translated_x * math.cos(angle_radians) + translated_y * math.sin(
        angle_radians
    )
    rotated_y = translated_y * math.cos(angle_radians) - translated_x * math.sin(
        angle_radians
    )

    # Translate back
    final_x = rotated_x + center_x
    final_y = rotated_y + center_y

    return (final_x, final_y)


def rotate_point_counterclockwise(
    point: Union[np.ndarray, Tuple[float, float]],
    center: Union[np.ndarray, Tuple[float, float]],
    angle_degrees: float,
) -> Tuple[float, float]:
    """
    Rotate a point counter-clockwise around a center point

    Parameters:
    -----------
    point : array-like
        Point to rotate
    center : array-like
        Center of rotation
    angle_degrees : float
        Rotation angle in degrees

    Returns:
    --------
    tuple
        Rotated point coordinates
    """
    x, y = point
    center_x, center_y = center
    angle_radians = math.radians(angle_degrees)

    # Translate point to origin
    translated_x = x - center_x
    translated_y = y - center_y

    # Rotate
    rotated_x = translated_x * math.cos(angle_radians) - translated_y * math.sin(
        angle_radians
    )
    rotated_y = translated_x * math.sin(angle_radians) + translated_y * math.cos(
        angle_radians
    )

    # Translate back
    final_x = rotated_x + center_x
    final_y = rotated_y + center_y

    return (final_x, final_y)
