import math
from typing import Tuple, Union

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


def create_line_equation(
    point1: np.ndarray,
    point2: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Create a line equation in the form Ax + By + C = 0

    Parameters:
    -----------
    point1, point2 : array-like
        Two points defining the line

    Returns:
    --------
    tuple
        Coefficients (A, B, C) where Ax + By + C = 0
    """
    A = point1[1] - point2[1]
    B = point2[0] - point1[0]
    C = point1[0] * point2[1] - point2[0] * point1[1]
    return A, B, -C


def calculate_line_intersection(
    line1: Tuple[float, float, float], line2: Tuple[float, float, float]
) -> Union[Tuple[float, float], None]:
    """
    Calculate intersection point of two lines

    Parameters:
    -----------
    line1, line2 : tuple
        Line coefficients (A, B, C) where Ax + By + C = 0

    Returns:
    --------
    tuple or None
        Coordinates of intersection point or None if lines are parallel
    """
    D = line1[0] * line2[1] - line1[1] * line2[0]
    Dx = line1[2] * line2[1] - line1[1] * line2[2]
    Dy = line1[0] * line2[2] - line1[2] * line2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return None


def calculate_parallel_line_distance(
    line1: Tuple[float, float, float], line2: Tuple[float, float, float]
) -> float:
    """
    Calculate the distance between two parallel lines

    Parameters:
    -----------
    line1, line2 : tuple
        Line coefficients (A, B, C) where Ax + By + C = 0

    Returns:
    --------
    float
        Distance between lines
    """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    eps = 1e-10

    # Normalize equations to the form: x + (B/A)y + (C/A) = 0
    new_C1 = C1 / (A1 + eps)

    new_A2 = 1
    new_B2 = B2 / (A2 + eps)
    new_C2 = C2 / (A2 + eps)

    # Calculate distance using the formula for parallel lines
    distance = (np.abs(new_C1 - new_C2)) / (np.sqrt(new_A2 * new_A2 + new_B2 * new_B2))
    return distance


def project_point_to_line(
    point_x: float,
    point_y: float,
    line_x1: float,
    line_y1: float,
    line_x2: float,
    line_y2: float,
) -> Tuple[float, float]:
    """
    Project a point onto a line

    Parameters:
    -----------
    point_x, point_y : float
        Coordinates of the point to project
    line_x1, line_y1, line_x2, line_y2 : float
        Coordinates of two points defining the line

    Returns:
    --------
    tuple
        Coordinates of the projected point
    """
    # Calculate the projected point coordinates
    eps = 1e-10
    x = (
        point_x * (line_x2 - line_x1) * (line_x2 - line_x1)
        + point_y * (line_y2 - line_y1) * (line_x2 - line_x1)
        + (line_x1 * line_y2 - line_x2 * line_y1) * (line_y2 - line_y1)
    ) / (
        (
            (line_x2 - line_x1) * (line_x2 - line_x1)
            + (line_y2 - line_y1) * (line_y2 - line_y1)
        )
        + eps
    )

    y = (
        point_x * (line_x2 - line_x1) * (line_y2 - line_y1)
        + point_y * (line_y2 - line_y1) * (line_y2 - line_y1)
        + (line_x2 * line_y1 - line_x1 * line_y2) * (line_x2 - line_x1)
    ) / (
        (
            (line_x2 - line_x1) * (line_x2 - line_x1)
            + (line_y2 - line_y1) * (line_y2 - line_y1)
        )
        + eps
    )

    return (x, y)


def rotate_point(
    point: np.ndarray,
    center: np.ndarray,
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
