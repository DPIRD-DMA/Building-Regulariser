import math

import numpy as np


def calculate_distance(
    point_1: np.ndarray,
    point_2: np.ndarray,
) -> float:
    """
    Calculate Euclidean distance between two points.

    Parameters:
    -----------
    point_1 : array-like
        First point coordinates
    point_2 : array-like
        Second point coordinates

    Returns:
    --------
    float
        Euclidean distance
    """
    return np.sqrt(np.sum(np.power((point_1 - point_2), 2)))


def calculate_azimuth_angle(
    start_point: np.ndarray,
    end_point: np.ndarray,
) -> float:
    """
    Calculate azimuth angle of line between two points (in degrees).

    Returns angle with respect to horizontal axis (x-axis).

    Parameters:
    -----------
    start_point : array-like
        Starting point coordinates
    end_point : array-like
        Ending point coordinates

    Returns:
    --------
    float
        Angle in degrees
    """
    # Ensure we have proper coordinates by converting to tuples of floats
    if hasattr(start_point, "__iter__"):
        x1, y1 = float(start_point[0]), float(start_point[1])
    else:
        raise TypeError("start_point must be a sequence with at least 2 elements")

    if hasattr(end_point, "__iter__"):
        x2, y2 = float(end_point[0]), float(end_point[1])
    else:
        raise TypeError("end_point must be a sequence with at least 2 elements")

    if x1 < x2:
        if y1 < y2:
            angle = math.atan((y2 - y1) / (x2 - x1))
            angle_degrees = angle * 180 / math.pi
            return angle_degrees
        elif y1 > y2:
            angle = math.atan((y1 - y2) / (x2 - x1))
            angle_degrees = angle * 180 / math.pi
            return 90 + (90 - angle_degrees)
        else:  # y1 == y2
            return 0
    elif x1 > x2:
        if y1 < y2:
            angle = math.atan((y2 - y1) / (x1 - x2))
            angle_degrees = angle * 180 / math.pi
            return 90 + (90 - angle_degrees)
        elif y1 > y2:
            angle = math.atan((y1 - y2) / (x1 - x2))
            angle_degrees = angle * 180 / math.pi
            return angle_degrees
        else:  # y1 == y2
            return 0
    else:  # x1 == x2
        return 90
