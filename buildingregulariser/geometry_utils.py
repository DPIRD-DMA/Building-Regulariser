import math
from typing import List, Tuple, Union

import numpy as np


def calculate_distance(
    point_1: np.ndarray,
    point_2: np.ndarray,
) -> float:
    """
    Calculate Euclidean distance between two points.

    Parameters:
    -----------
    point_1 : np.ndarray
        First point coordinates
    point_2 : np.ndarray
        Second point coordinates

    Returns:
    --------
    float
        Euclidean distance
    """
    # return np.sqrt(np.sum(np.power((point_1 - point_2), 2)))
    dx, dy = point_1[0] - point_2[0], point_1[1] - point_2[1]
    return math.hypot(dx, dy)  # fastest for scalar calls


def calculate_azimuth_angle(start_point: np.ndarray, end_point: np.ndarray) -> float:
    """
    Calculate azimuth angle of the line from start_point to end_point (in degrees).
    Angle is measured clockwise from the positive x-axis.

    Parameters:
    -----------
    start_point : np.ndarray
        Starting point coordinates
    end_point : np.ndarray
        Ending point coordinates

    Returns:
    --------
    float
        Angle in degrees in the range [0, 360)
    """
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees % 360


def create_line_equation(
    point1: np.ndarray,
    point2: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Create a line equation in the form Ax + By + C = 0

    Parameters:
    -----------
    point1, point2 : np.ndarray
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
    line1: Tuple[float, float, float],
    line2: Tuple[float, float, float],
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
    line1: Tuple[float, float, float],
    line2: Tuple[float, float, float],
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
    A1, _, C1 = line1
    A2, B2, C2 = line2
    eps = 1e-10

    # Normalize equations to the form: x + (B/A)y + (C/A) = 0
    new_C1 = C1 / (A1 + eps)
    new_A2 = 1
    new_B2 = B2 / (A2 + eps)
    new_C2 = C2 / (A2 + eps)

    # Calculate distance using the formula for parallel lines
    distance = abs(new_C1 - new_C2) / math.sqrt(new_A2 * new_A2 + new_B2 * new_B2)
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
    Project a point onto a line.

    Parameters:
    -----------
    point_x, point_y : float
        Coordinates of the point to project
    line_x1, line_y1, line_x2, line_y2 : float
        Coordinates of two points defining the line

    Returns:
    --------
    Tuple[float, float]
        Coordinates of the projected point
    """
    eps = 1e-10
    dx = line_x2 - line_x1
    dy = line_y2 - line_y1
    denom = dx * dx + dy * dy + eps

    x = (
        point_x * dx * dx
        + point_y * dy * dx
        + (line_x1 * line_y2 - line_x2 * line_y1) * dy
    ) / denom

    y = (
        point_x * dx * dy
        + point_y * dy * dy
        + (line_x2 * line_y1 - line_x1 * line_y2) * dx
    ) / denom

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
    point : np.ndarray
        Point to rotate
    center : np.ndarray
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


def rotate_edge(
    start_point: np.ndarray, end_point: np.ndarray, rotation_angle: float
) -> List[np.ndarray]:
    """
    Rotate an edge around its midpoint by the given angle

    Parameters:
    -----------
    start_point : numpy.ndarray
        Start point of the edge
    end_point : numpy.ndarray
        End point of the edge
    rotation_angle : float
        Angle to rotate by in degrees

    Returns:
    --------
    list
        List containing the rotated start and end points
    """
    midpoint = (start_point + end_point) / 2

    if rotation_angle > 0:
        rotated_start = rotate_point(start_point, midpoint, -rotation_angle)
        rotated_end = rotate_point(end_point, midpoint, -rotation_angle)
    elif rotation_angle < 0:
        rotated_start = rotate_point(start_point, midpoint, np.abs(rotation_angle))
        rotated_end = rotate_point(end_point, midpoint, np.abs(rotation_angle))
    else:
        rotated_start = start_point
        rotated_end = end_point

    return [np.array(rotated_start), np.array(rotated_end)]
