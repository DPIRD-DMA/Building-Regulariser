from typing import Tuple, Union

import numpy as np


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
