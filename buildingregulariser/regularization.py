"""
Core regularization algorithms for polygon simplification.

Functions for regularizing coordinate arrays and polygons.
"""

import warnings
from typing import List

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .geometry_utils import calculate_azimuth_angle, calculate_distance
from .line_operations import (
    calculate_line_intersection,
    calculate_parallel_line_distance,
    create_line_equation,
    project_point_to_line,
)
from .rotation import rotate_point_clockwise, rotate_point_counterclockwise
from .simplification import simplify_with_rdp


def regularize_coordinate_array(
    coordinates: np.ndarray, epsilon: float = 6, parallel_threshold: float = 3
) -> np.ndarray:
    """
    Regularize polygon coordinates by aligning edges to be either parallel
    or perpendicular to the longest edge

    Parameters:
    -----------
    coordinates : numpy.ndarray
        Array of coordinates for a polygon ring (shape: n x 2)
    epsilon : float
        Parameter for Ramer-Douglas-Peucker algorithm (higher = more simplification)
    parallel_threshold : float
        Distance threshold for considering parallel lines as needing connection

    Returns:
    --------
    numpy.ndarray
        Regularized coordinates array
    """
    # Ensure coordinates are closed
    if not np.array_equal(coordinates[0], coordinates[-1]):
        coordinates = np.vstack([coordinates, coordinates[0]])

    # Simplify using RDP algorithm
    simplified_coordinates = simplify_with_rdp(coordinates, epsilon=epsilon)

    # Ensure simplified_coordinates is a numpy array
    if not isinstance(simplified_coordinates, np.ndarray):
        simplified_coordinates = np.array(simplified_coordinates)

    # Analyze edges to find properties and main direction
    edge_data = analyze_edges(simplified_coordinates)

    # Orient edges based on longest edge direction
    oriented_edges, edge_orientations = orient_edges(simplified_coordinates, edge_data)

    # Connect and regularize edges
    regularized_points = connect_regularized_edges(
        oriented_edges, edge_orientations, parallel_threshold
    )

    return np.array(regularized_points, dtype=object)


def analyze_edges(simplified_coordinates: np.ndarray):
    """
    Analyze edges to find their lengths, angles, and indices

    Parameters:
    -----------
    simplified_coordinates : numpy.ndarray
        Simplified polygon coordinates

    Returns:
    --------
    dict
        Dictionary containing edge data including lengths, angles, indices, and main direction
    """
    edge_lengths = []
    azimuth_angles = []
    edge_indices = []

    for i in range(len(simplified_coordinates) - 1):
        current_point = simplified_coordinates[i]
        next_point = simplified_coordinates[i + 1]

        edge_length = calculate_distance(current_point, next_point)
        azimuth = calculate_azimuth_angle(current_point, next_point)

        edge_lengths.append(edge_length)
        azimuth_angles.append(azimuth)
        edge_indices.append([i, i + 1])

    # Find longest edge and use its direction as main direction
    longest_edge_index = np.argmax(edge_lengths)
    main_direction = azimuth_angles[longest_edge_index]

    return {
        "edge_lengths": edge_lengths,
        "azimuth_angles": azimuth_angles,
        "edge_indices": edge_indices,
        "longest_edge_index": longest_edge_index,
        "main_direction": main_direction,
    }


def orient_edges(simplified_coordinates: np.ndarray, edge_data: dict):
    """
    Orient edges to be parallel or perpendicular to the main direction

    Parameters:
    -----------
    simplified_coordinates : numpy.ndarray
        Simplified polygon coordinates
    edge_data : dict
        Dictionary containing edge analysis data

    Returns:
    --------
    tuple
        Tuple containing oriented edges array and edge orientations list
    """
    oriented_edges = []
    edge_orientations = []  # 0=parallel, 1=perpendicular

    azimuth_angles = edge_data["azimuth_angles"]
    edge_indices = edge_data["edge_indices"]
    longest_edge_index = edge_data["longest_edge_index"]
    main_direction = edge_data["main_direction"]

    for i, (azimuth, (start_idx, end_idx)) in enumerate(
        zip(azimuth_angles, edge_indices)
    ):
        if i == longest_edge_index:
            # Keep longest edge as is
            oriented_edges.append(
                [
                    np.array(simplified_coordinates[start_idx]),
                    np.array(simplified_coordinates[end_idx]),
                ]
            )
            edge_orientations.append(0)  # Parallel to main direction
        else:
            # Determine rotation angle to align with main direction
            rotation_angle = main_direction - azimuth

            if np.abs(rotation_angle) < 45:  # 180/4 = 45
                # Make parallel to main direction
                edge_orientations.append(0)
            elif np.abs(rotation_angle) >= 45:
                # Make perpendicular to main direction
                rotation_angle = rotation_angle + 90
                edge_orientations.append(1)

            # Perform rotation
            start_point = np.array(simplified_coordinates[start_idx], dtype=float)
            end_point = np.array(simplified_coordinates[end_idx], dtype=float)
            rotated_edge = rotate_edge(start_point, end_point, rotation_angle)

            oriented_edges.append(rotated_edge)

    return np.array(oriented_edges, dtype=object), edge_orientations


def rotate_edge(start_point: np.ndarray, end_point: np.ndarray, rotation_angle: float):
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
        rotated_start = rotate_point_counterclockwise(
            start_point, midpoint, np.abs(rotation_angle)
        )
        rotated_end = rotate_point_counterclockwise(
            end_point, midpoint, np.abs(rotation_angle)
        )
    elif rotation_angle < 0:
        rotated_start = rotate_point_clockwise(
            start_point, midpoint, np.abs(rotation_angle)
        )
        rotated_end = rotate_point_clockwise(
            end_point, midpoint, np.abs(rotation_angle)
        )
    else:
        rotated_start = start_point
        rotated_end = end_point

    return [np.array(rotated_start), np.array(rotated_end)]


def connect_regularized_edges(
    oriented_edges: np.ndarray, edge_orientations: list, parallel_threshold: float
):
    """
    Connect oriented edges to form a regularized polygon

    Parameters:
    -----------
    oriented_edges : numpy.ndarray
        Array of oriented edges
    edge_orientations : list
        List of edge orientations (0=parallel, 1=perpendicular)
    parallel_threshold : float
        Distance threshold for considering parallel lines as needing connection

    Returns:
    --------
    list
        List of regularized points forming the polygon
    """
    regularized_points = []
    regularized_points.append(oriented_edges[0][0])

    for i in range(len(oriented_edges) - 1):
        current_index = i
        next_index = i + 1 if i < len(oriented_edges) - 1 else 0

        current_edge_start = oriented_edges[current_index][0]
        current_edge_end = oriented_edges[current_index][1]
        next_edge_start = oriented_edges[next_index][0]
        next_edge_end = oriented_edges[next_index][1]

        current_orientation = edge_orientations[current_index]
        next_orientation = edge_orientations[next_index]

        if current_orientation != next_orientation:
            # Handle perpendicular edges
            regularized_points.append(
                handle_perpendicular_edges(
                    current_edge_start, current_edge_end, next_edge_start, next_edge_end
                )
            )
        else:
            # Handle parallel edges
            new_points = handle_parallel_edges(
                current_edge_start,
                current_edge_end,
                next_edge_start,
                next_edge_end,
                parallel_threshold,
                next_index,
                oriented_edges,
            )
            regularized_points.extend(new_points)

    # Close the polygon
    if len(regularized_points) > 0 and not np.array_equal(
        regularized_points[0], regularized_points[-1]
    ):
        regularized_points.append(regularized_points[0])

    return regularized_points


def handle_perpendicular_edges(
    current_edge_start, current_edge_end, next_edge_start, next_edge_end
):
    """
    Handle intersection of perpendicular edges

    Parameters:
    -----------
    current_edge_start : numpy.ndarray
        Start point of current edge
    current_edge_end : numpy.ndarray
        End point of current edge
    next_edge_start : numpy.ndarray
        Start point of next edge
    next_edge_end : numpy.ndarray
        End point of next edge

    Returns:
    --------
    numpy.ndarray
        Intersection point of the two edges
    """
    line1 = create_line_equation(current_edge_start, current_edge_end)
    line2 = create_line_equation(next_edge_start, next_edge_end)

    intersection_point = calculate_line_intersection(line1, line2)
    if intersection_point:
        # Convert to numpy array if not already
        return np.array(intersection_point)
    else:
        # If lines are parallel (shouldn't happen with perpendicular check)
        # add the end point of current edge
        return current_edge_end


def handle_parallel_edges(
    current_edge_start,
    current_edge_end,
    next_edge_start,
    next_edge_end,
    parallel_threshold,
    next_index,
    oriented_edges,
):
    """
    Handle connection between parallel edges

    Parameters:
    -----------
    current_edge_start : numpy.ndarray
        Start point of current edge
    current_edge_end : numpy.ndarray
        End point of current edge
    next_edge_start : numpy.ndarray
        Start point of next edge
    next_edge_end : numpy.ndarray
        End point of next edge
    parallel_threshold : float
        Distance threshold for considering parallel lines as needing connection
    next_index : int
        Index of the next edge
    oriented_edges : numpy.ndarray
        Array of all oriented edges

    Returns:
    --------
    list
        List of points to add to the regularized polygon
    """
    line1 = create_line_equation(current_edge_start, current_edge_end)
    line2 = create_line_equation(next_edge_start, next_edge_end)
    line_distance = calculate_parallel_line_distance(line1, line2)

    new_points = []

    if line_distance < parallel_threshold:
        # Shift next edge to align with current edge
        projected_point = project_point_to_line(
            next_edge_start[0],
            next_edge_start[1],
            current_edge_start[0],
            current_edge_start[1],
            current_edge_end[0],
            current_edge_end[1],
        )
        # Ensure projected_point is a numpy array
        new_points.append(np.array(projected_point))

        # Update next edge starting point
        oriented_edges[next_index][0] = np.array(projected_point)
        oriented_edges[next_index][1] = np.array(
            project_point_to_line(
                next_edge_end[0],
                next_edge_end[1],
                current_edge_start[0],
                current_edge_start[1],
                current_edge_end[0],
                current_edge_end[1],
            )
        )
    else:
        # Add connecting segment between edges
        midpoint = (current_edge_end + next_edge_start) / 2
        connecting_point1 = project_point_to_line(
            midpoint[0],
            midpoint[1],
            current_edge_start[0],
            current_edge_start[1],
            current_edge_end[0],
            current_edge_end[1],
        )
        connecting_point2 = project_point_to_line(
            midpoint[0],
            midpoint[1],
            next_edge_start[0],
            next_edge_start[1],
            next_edge_end[0],
            next_edge_end[1],
        )
        # Convert points to numpy arrays
        new_points.append(np.array(connecting_point1))
        new_points.append(np.array(connecting_point2))

    return new_points


def poly_poly_iou(poly_1: Polygon, poly_2: Polygon) -> float:
    """
    Calculate the Intersection over Union (IoU) between two polygons

    Parameters:
    -----------
    poly_1 : shapely.geometry.Polygon
        First polygon
    poly_2 : shapely.geometry.Polygon
        Second polygon

    Returns:
    --------
    float
        IoU value between 0 and 1
    """
    if poly_1.is_empty or poly_2.is_empty:
        return 0.0
    if not poly_1.is_valid or not poly_2.is_valid:
        return 0.0
    if not poly_1.intersects(poly_2):
        return 0.0
    intersection_area = poly_1.intersection(poly_2).area
    union_area = poly_1.union(poly_2).area
    return intersection_area / union_area if union_area > 0 else 0.0


def hausdorff_ratio(poly1, poly2):
    coords1 = np.array(poly1.exterior.coords)
    coords2 = np.array(poly2.exterior.coords)

    forward = directed_hausdorff(coords1, coords2)[0]
    backward = directed_hausdorff(coords2, coords1)[0]

    return max(forward, backward)


def regularize_single_polygon(
    polygon: Polygon,
    epsilon: float = 6,
    parallel_threshold: float = 3,
) -> Polygon:
    """
    Regularize a Shapely polygon by aligning edges to principal directions

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
        Input polygon to regularize
    epsilon : float
        Parameter for Ramer-Douglas-Peucker algorithm (higher = more simplification)
    parallel_threshold : float
        Distance threshold for parallel line handling

    Returns:
    --------
    shapely.geometry.Polygon
        Regularized polygon
    """

    # Handle exterior ring

    exterior_coordinates = np.array(polygon.exterior.coords)
    regularized_exterior = regularize_coordinate_array(
        exterior_coordinates, epsilon=epsilon, parallel_threshold=parallel_threshold
    )

    # Handle interior rings (holes)
    regularized_interiors: List[np.ndarray] = []
    for interior in polygon.interiors:
        interior_coordinates = np.array(interior.coords)
        regularized_interior = regularize_coordinate_array(
            interior_coordinates, epsilon=epsilon, parallel_threshold=parallel_threshold
        )
        regularized_interiors.append(regularized_interior)

    # Create new polygon
    try:
        # Convert coordinates to LinearRings
        exterior_ring = LinearRing(regularized_exterior)
        interior_rings = [LinearRing(r) for r in regularized_interiors]

        # Create regularized polygon
        regularized_polygon = Polygon(exterior_ring, interior_rings)
        return regularized_polygon
    except Exception as e:
        # If there's an error creating the polygon, return the original
        warnings.warn(f"Error creating regularized polygon: {e}. Returning original.")
        return polygon


def process_geometry(
    geometry: BaseGeometry, epsilon: float, parallel_threshold: float
) -> BaseGeometry:
    """
    Process a single geometry, handling different geometry types

    Parameters:
    -----------
    geometry : shapely.geometry.BaseGeometry
        Input geometry to regularize
    epsilon : float
        Parameter for Ramer-Douglas-Peucker algorithm
    parallel_threshold : float
        Distance threshold for parallel line handling

    Returns:
    --------
    shapely.geometry.BaseGeometry
        Regularized geometry
    """
    if isinstance(geometry, Polygon):
        return regularize_single_polygon(geometry, epsilon, parallel_threshold)
    elif isinstance(geometry, MultiPolygon):
        regularized_parts = [
            regularize_single_polygon(part, epsilon, parallel_threshold)
            for part in geometry.geoms
        ]
        return MultiPolygon(regularized_parts)
    else:
        # Return unmodified if not a polygon
        warnings.warn(
            f"Unsupported geometry type: {type(geometry)}. Returning original."
        )
        return geometry
