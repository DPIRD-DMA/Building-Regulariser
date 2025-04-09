import warnings
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LinearRing, Polygon
from shapely.geometry.base import BaseGeometry

from .geometry_utils import calculate_azimuth_angle, calculate_distance
from .line_operations import (
    calculate_line_intersection,
    calculate_parallel_line_distance,
    create_line_equation,
    project_point_to_line,
)
from .rotation import rotate_point


def find_nearest_target_angle(
    current_azimuth: float, main_direction: float, allow_45_degree: bool
) -> float:
    """
    Finds the closest allowed target azimuth angle (0-360).
    """
    # Calculate angular difference relative to main_direction, normalize to [-180, 180]
    diff_angle = (current_azimuth - main_direction + 180) % 360 - 180

    # Define potential offsets from the main direction
    allowed_offsets = []
    if allow_45_degree:
        # Use offsets like 0, 45, 90, 135, 180, -45, -90, -135
        # Note: 180 and -180 are equivalent, 225 is -135, 270 is -90, 315 is -45
        allowed_offsets = [0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0]
    else:
        # Use offsets 0, 90, 180, -90 (or 270)
        allowed_offsets = [0.0, 90.0, 180.0, -90.0]

    # Find the offset that minimizes the absolute difference to diff_angle
    best_offset = 0.0
    min_angle_dist = 181.0  # Start with a value larger than max possible diff (180)

    for offset in allowed_offsets:
        # Calculate the shortest angle between diff_angle and the current offset
        d = (diff_angle - offset + 180) % 360 - 180
        if abs(d) < min_angle_dist:
            min_angle_dist = abs(d)
            best_offset = offset

    # Calculate the final target azimuth by adding the best offset to the main direction
    # Normalize to [0, 360)
    target_azimuth = (main_direction + best_offset + 360) % 360
    return target_azimuth


def enforce_angles_post_process(
    points: List[np.ndarray],
    main_direction: float,
    allow_45_degree: bool,
    angle_tolerance: float = 0.1,  # Tolerance in degrees
    max_iterations: int = 2,  # Number of passes to allow adjustments to settle
) -> List[np.ndarray]:
    """
    Adjusts vertices iteratively to enforce target angles for each segment.
    Runs multiple iterations as adjusting one segment can affect adjacent ones.

    Parameters:
    -----------
    points : list[np.ndarray]
        List of numpy arrays representing polygon vertices. Assumed NOT closed
        (last point != first point). Length N >= 3.
    main_direction : float
        The main direction angle in degrees (0-360).
    allow_45_degree : bool
        Whether to allow 45-degree angles.
    angle_tolerance : float
         Allowable deviation from target angle in degrees.
    max_iterations : int
         Maximum number of full passes to adjust angles.

    Returns:
    --------
    list[np.ndarray]
        List of adjusted vertices (N points).
    """
    if len(points) < 3:
        return points  # Not enough points to form segments

    adjusted_points = [p.copy() for p in points]  # Work on a copy
    num_points = len(adjusted_points)

    for _ in range(max_iterations):
        max_angle_diff_this_iter = 0.0
        changed = False  # Flag to track if any changes were made in this iteration

        for i in range(num_points):
            p1_idx = i
            p2_idx = (i + 1) % num_points  # Wrap around for the end point index

            p1 = adjusted_points[p1_idx]
            p2 = adjusted_points[p2_idx]

            # Avoid issues with coincident points before calculating angle
            dist = calculate_distance(p1, p2)
            if dist < 1e-7:
                # Coincident points have undefined angle, skip adjustment for this segment
                continue

            current_azimuth = calculate_azimuth_angle(p1, p2)
            target_azimuth = find_nearest_target_angle(
                current_azimuth, main_direction, allow_45_degree
            )

            # Calculate shortest rotation angle needed (positive for counter-clockwise)
            rotation_diff = (target_azimuth - current_azimuth + 180) % 360 - 180

            # Track the maximum deviation found in this iteration
            max_angle_diff_this_iter = max(max_angle_diff_this_iter, abs(rotation_diff))

            # Only rotate if the difference significantly exceeds tolerance
            # Use a slightly larger threshold for making changes to prevent jitter
            if abs(rotation_diff) > angle_tolerance:
                changed = True  # Mark that an adjustment was made

                # Perform rotation (rotation_diff > 0 means counter-clockwise)
                if rotation_diff > 0:
                    new_p2_tuple = rotate_point(p2, p1, -rotation_diff)
                else:
                    new_p2_tuple = rotate_point(p2, p1, abs(rotation_diff))

                # Update the endpoint in the list for the *next* segment's calculation
                adjusted_points[p2_idx] = np.array(new_p2_tuple)

        # Check for convergence: If no points were adjusted significantly in this pass, stop.
        if not changed:
            break

    # Return the list of N adjusted unique points
    return adjusted_points


def regularize_coordinate_array(
    coordinates: np.ndarray,
    parallel_threshold: float = 3,
    allow_45_degree: bool = True,
    diagonal_threshold_reduction: float = 15.0,
    angle_enforcement_tolerance: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """
    Regularize polygon coordinates by aligning edges to be either parallel
    or perpendicular (or 45 deg) to the main direction, with a
    post-processing step to enforce angles.

    Parameters:
    -----------
    coordinates : numpy.ndarray
        Array of coordinates for a polygon ring (shape: n x 2).
        Assumed closed (first point == last point).
    parallel_threshold : float
        Distance threshold for considering parallel lines as needing connection.
    allow_45_degree : bool
        If True, allows 45-degree orientations relative to the main direction.
    angle_enforcement_tolerance : float
        Maximum allowed deviation (degrees) from target angle in the final output.

    Returns:
    --------
    numpy.ndarray
        Regularized coordinates array (n x 2), closed (first == last).
    """
    if (
        len(coordinates) < 4
    ):  # Need at least 3 unique points + closing point for a polygon
        warnings.warn("Not enough coordinates to regularize. Returning original.")
        return coordinates, 0.0

    # Remove duplicate closing point for processing, if present
    if np.allclose(coordinates[0], coordinates[-1]):
        processing_coords = coordinates[:-1]
    else:
        processing_coords = coordinates  # Assume it wasn't closed

    if len(processing_coords) < 3:
        warnings.warn(
            "Not enough unique coordinates to regularize. Returning original."
        )
        return coordinates, 0.0  # Return original closed coords

    # Analyze edges to find properties and main direction
    # Use the non-closed version for edge analysis
    edge_data = analyze_edges(processing_coords)

    # Orient edges based on main direction
    oriented_edges, edge_orientations = orient_edges(
        processing_coords,
        edge_data,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
    )

    # Connect and regularize edges
    # This returns a list of np.ndarray points, likely NOT closed
    initial_regularized_points = connect_regularized_edges(
        oriented_edges, edge_orientations, parallel_threshold
    )

    if not initial_regularized_points or len(initial_regularized_points) < 3:
        warnings.warn("Regularization resulted in too few points. Returning original.")
        # Decide what to return: maybe the simplified input or original?
        # Returning original for safety:
        return coordinates, 0.0

    final_regularized_points_list = enforce_angles_post_process(
        initial_regularized_points,
        edge_data["main_direction"],
        allow_45_degree,
        angle_tolerance=angle_enforcement_tolerance,
    )

    if not final_regularized_points_list or len(final_regularized_points_list) < 3:
        warnings.warn(
            "Angle enforcement resulted in too few points. Returning original."
        )
        return coordinates, 0.0

    # Convert list of arrays back to a single numpy array and ensure closure
    final_coords_array = np.array([p for p in final_regularized_points_list])
    # Ensure the final array is explicitly closed for Shapely
    closed_final_coords = np.vstack([final_coords_array, final_coords_array[0]])

    return closed_final_coords, edge_data["main_direction"]


def analyze_edges(
    coordinates: np.ndarray, coarse_bin_size: int = 5, fine_bin_size: int = 1
):
    """
    Analyze edges to determine azimuth angles and main structural direction.

    Parameters:
    -----------
    coordinates : np.ndarray
        Polygon coordinates (shape: N x 2), assumed NOT closed.

    Returns:
    --------
    dict
        Dictionary containing:
        - azimuth_angles: array of absolute edge angles (degrees)
        - edge_indices: array of [start_idx, end_idx] pairs for each edge
        - main_direction: float angle (degrees) representing dominant structure orientation
    """
    if len(coordinates) < 3:
        return {
            "azimuth_angles": np.array([]),
            "edge_indices": np.array([]),
            "main_direction": 0,
        }

    def create_weighted_histogram(
        angles: np.ndarray, bin_size: float, weights: np.ndarray, num_bins_override=None
    ):
        num_bins = (
            int(90 / bin_size) if num_bins_override is None else num_bins_override
        )
        indices = np.minimum(np.floor(angles / bin_size).astype(int), num_bins - 1)
        return np.bincount(indices, weights=weights, minlength=num_bins)

    def dist_to_dir(d: float) -> np.ndarray:
        return np.minimum.reduce(
            [
                np.abs(normalized_angles - d),
                np.abs(normalized_angles - d - 180),
                np.abs(normalized_angles - d + 180),
            ]
        )

    # Form edges and compute vectors
    start_points = coordinates
    end_points = np.roll(coordinates, -1, axis=0)
    vectors = end_points - start_points
    edge_lengths = np.linalg.norm(vectors, axis=1)

    # Filter out very short edges
    valid = edge_lengths > 1e-9
    if not np.any(valid):
        return {
            "azimuth_angles": np.array([]),
            "edge_indices": np.array([]),
            "main_direction": 0,
        }

    vectors = vectors[valid]
    lengths = edge_lengths[valid]
    azimuth_angles = (np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) + 360) % 360
    normalized_angles = azimuth_angles % 180
    orthogonal_angles = normalized_angles % 90

    # Store edge indices
    indices = np.stack(
        [
            np.arange(len(coordinates)),
            (np.arange(len(coordinates)) + 1) % len(coordinates),
        ],
        axis=1,
    )
    edge_indices = indices[valid]

    coarse_bins = create_weighted_histogram(orthogonal_angles, coarse_bin_size, lengths)
    fine_bins = create_weighted_histogram(
        orthogonal_angles, fine_bin_size, lengths, num_bins_override=90
    )

    if np.sum(coarse_bins) == 0:
        main_direction = 0
    else:
        # Step 1: Coarse dominant bin
        main_bin = np.argmax(coarse_bins)
        fine_start = main_bin * coarse_bin_size
        fine_end = fine_start + coarse_bin_size

        # Step 2: Refine with fine bin
        refined_bin = np.argmax(fine_bins[fine_start:fine_end])
        refined_angle = fine_start + refined_bin + fine_bin_size / 2

        # Step 3: Choose between refined or perpendicular angle
        dir1 = refined_angle
        dir2 = (refined_angle + 90) % 180

        support1 = np.sum(lengths[dist_to_dir(dir1) <= dist_to_dir(dir2)])
        support2 = np.sum(lengths[dist_to_dir(dir1) > dist_to_dir(dir2)])

        main_direction = dir1 if support1 >= support2 else dir2

    return {
        "azimuth_angles": azimuth_angles,
        "edge_indices": edge_indices,
        "main_direction": main_direction,
    }


def orient_edges(
    simplified_coordinates: np.ndarray,
    edge_data: dict,
    allow_45_degree: bool = True,
    diagonal_threshold_reduction: float = 15.0,
):
    """
    Orient edges to be parallel or perpendicular (or optionally 45 degrees)
    to the main direction determined by angle distribution analysis.

    Parameters:
    -----------
    simplified_coordinates : numpy.ndarray
        Simplified polygon coordinates (shape: n x 2, assumed closed).
    edge_data : dict
        Dictionary containing edge analysis data ('azimuth_angles', 'edge_indices',
        'main_direction').
    allow_45_degree : bool, optional
        If True, allows edges to be oriented at 45-degree angles relative
        to the main direction. Defaults to True.
    diagonal_threshold_reduction : float, optional
        Angle in degrees to subtract from the 45-degree snapping thresholds,
        making diagonal (45°) orientations less likely. Defaults to 15.0.

    Returns:
    --------
    tuple
        Tuple containing:
        - oriented_edges (numpy.ndarray): Array of [start, end] points for each oriented edge.
        - edge_orientations (list): List of orientation codes for each edge.
          - 0: Parallel or anti-parallel (0, 180 deg relative to main_direction)
          - 1: Perpendicular (90, 270 deg relative to main_direction)
          - 2: Diagonal (45, 135, 225, 315 deg relative to main_direction) - only if allow_45=True.
    """

    # edge_data =
    oriented_edges = []
    # Orientation codes: 0=Parallel/AntiParallel, 1=Perpendicular, 2=Diagonal(45deg)
    edge_orientations = []

    azimuth_angles = edge_data["azimuth_angles"]
    edge_indices = edge_data["edge_indices"]
    main_direction = edge_data["main_direction"]

    # Small tolerance for floating point comparisons
    tolerance = 1e-9

    for i, (azimuth, (start_idx, end_idx)) in enumerate(
        zip(azimuth_angles, edge_indices)
    ):
        # Calculate the shortest angle difference from edge azimuth to main_direction
        # Result is in the range [-180, 180]
        diff_angle = (azimuth - main_direction + 180) % 360 - 180

        target_offset = (
            0.0  # The desired angle relative to main_direction (0, 45, 90 etc.)
        )
        orientation_code = 0

        if allow_45_degree:
            # Calculate how close we are to each of the key orientations
            dist_to_0 = min(abs(diff_angle % 180), abs((diff_angle % 180) - 180))
            dist_to_90 = min(abs((diff_angle % 180) - 90), abs((diff_angle % 180) - 90))
            dist_to_45 = min(
                abs((diff_angle % 180) - 45), abs((diff_angle % 180) - 135)
            )

            # Apply down-weighting to 45-degree angles
            # This effectively shrinks the zone where angles snap to 45 degrees
            if dist_to_45 <= (22.5 - diagonal_threshold_reduction):
                # Close enough to 45/135/225/315 degrees (accounting for down-weighting)
                angle_mod = diff_angle % 90
                if angle_mod < 45:
                    target_offset = (diff_angle // 90) * 90 + 45
                else:
                    target_offset = (diff_angle // 90 + 1) * 90 - 45

                # Determine which diagonal direction we're closer to
                # Use modulo 180 to differentiate between 45/225 and 135/315
                normalized_angle = (main_direction + target_offset) % 180
                if 0 <= normalized_angle < 90:
                    # This is closer to 45 degrees
                    orientation_code = 2  # 45/225 degrees
                else:
                    # This is closer to 135 degrees
                    orientation_code = 3  # 135/315 degrees
            elif dist_to_0 <= dist_to_90:
                # Closer to 0/180 degrees
                target_offset = round(diff_angle / 180.0) * 180.0
                orientation_code = 0
            else:
                # Closer to 90/270 degrees
                target_offset = round(diff_angle / 90.0) * 90.0
                if abs(target_offset % 180) < tolerance:
                    # If rounding diff_angle/90 gave 0 or 180, force to 90 or -90
                    target_offset = 90.0 if diff_angle > 0 else -90.0
                orientation_code = 1

        else:  # Original logic (refined): Snap only to nearest 90 degrees
            if abs(diff_angle) < 45.0:  # Closer to parallel/anti-parallel (0 or 180)
                # Snap to 0 or 180, whichever is closer
                target_offset = round(diff_angle / 180.0) * 180.0
                orientation_code = 0
            else:  # Closer to perpendicular (+90 or -90/270)
                # Snap to +90 or -90, whichever is closer
                target_offset = round(diff_angle / 90.0) * 90.0
                # Ensure it's not actually 0 or 180 (should be handled above, but safety check)
                if abs(target_offset % 180) < tolerance:
                    # If rounding diff_angle/90 gave 0 or 180, force to 90 or -90
                    target_offset = 90.0 if diff_angle > 0 else -90.0
                orientation_code = 1

        # Calculate the rotation angle needed to achieve the target orientation
        # Calculate shortest rotation angle
        rotation_angle = (main_direction + target_offset - azimuth + 180) % 360 - 180

        # Perform rotation
        start_point = np.array(simplified_coordinates[start_idx], dtype=float)
        end_point = np.array(simplified_coordinates[end_idx], dtype=float)

        # Rotate the edge to align with the target orientation
        rotated_edge = rotate_edge(start_point, end_point, rotation_angle)

        oriented_edges.append(rotated_edge)
        edge_orientations.append(orientation_code)

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
        rotated_start = rotate_point(start_point, midpoint, -rotation_angle)
        rotated_end = rotate_point(end_point, midpoint, -rotation_angle)
    elif rotation_angle < 0:
        rotated_start = rotate_point(start_point, midpoint, np.abs(rotation_angle))
        rotated_end = rotate_point(end_point, midpoint, np.abs(rotation_angle))
    else:
        rotated_start = start_point
        rotated_end = end_point

    return [np.array(rotated_start), np.array(rotated_end)]


#     return regularized_points
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

    # Process all edges including the connection between last and first edge
    for i in range(len(oriented_edges)):
        current_index = i
        next_index = (i + 1) % len(oriented_edges)  # Wrap around to first edge

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


def regularize_single_polygon(
    polygon: Polygon,
    parallel_threshold: float = 3,
    allow_45_degree: bool = True,
    diagonal_threshold_reduction: float = 15.0,
    allow_circles: bool = True,
    circle_threshold: float = 0.90,
    include_metadata: bool = False,
) -> Tuple[Polygon, Optional[float], Optional[float]]:
    """
    Regularize a Shapely polygon by aligning edges to principal directions

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
        Input polygon to regularize
    parallel_threshold : float
        Distance threshold for parallel line handling
    allow_45_degree : bool
        If True, allows 45-degree orientations relative to the main direction
    allow_circles : bool
        If True, attempts to detect polygons that are nearly circular and
        replaces them with perfect circles
    circle_threshold : float
        Intersection over Union (IoU) threshold used for circle detection
        Value between 0 and 1
        Defaults to 0.9

    Returns:
    --------
    shapely.geometry.Polygon
        Regularized polygon
    """

    # Ensure the polygon is valid
    polygon = polygon.buffer(0)

    exterior_coordinates = np.array(polygon.exterior.coords)
    # append the first point to the end to close the polygon

    regularized_exterior, main_direction = regularize_coordinate_array(
        exterior_coordinates,
        parallel_threshold=parallel_threshold,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
    )

    if allow_circles:
        radius = np.sqrt(polygon.area / np.pi)
        perfect_circle = polygon.centroid.buffer(radius, resolution=42)
        # Check if the polygon is close to a circle using iou
        iou = (
            perfect_circle.intersection(polygon).area
            / perfect_circle.union(polygon).area
        )
        if iou > circle_threshold:
            # If the polygon is close to a circle, return the perfect circle
            regularized_exterior = np.array(
                perfect_circle.exterior.coords, dtype=object
            )

    # Handle interior rings (holes)
    regularized_interiors: List[np.ndarray] = []
    for interior in polygon.interiors:
        interior_coordinates = np.array(interior.coords)
        regularized_interior, _ = regularize_coordinate_array(
            interior_coordinates,
            parallel_threshold=parallel_threshold,
            allow_45_degree=allow_45_degree,
        )
        regularized_interiors.append(regularized_interior)

    # Create new polygon
    try:
        # Convert coordinates to LinearRings
        exterior_ring = LinearRing(regularized_exterior)
        interior_rings = [LinearRing(r) for r in regularized_interiors]

        # Create regularized polygon
        regularized_polygon = Polygon(exterior_ring, interior_rings).buffer(0)
        if include_metadata:
            final_iou = (
                regularized_polygon.intersection(polygon).area
                / regularized_polygon.union(polygon).area
            )
        else:
            final_iou = None

        return regularized_polygon, final_iou, main_direction
    except Exception as e:
        # If there's an error creating the polygon, return the original
        warnings.warn(f"Error creating regularized polygon: {e}. Returning original.")
        return polygon, None, None


def process_geometry(
    geometry: BaseGeometry,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    allow_circles: bool,
    circle_threshold: float,
    include_metadata: bool,
) -> Tuple[BaseGeometry, Optional[float], Optional[float]]:
    """
    Process a single geometry, handling different geometry types

    Parameters:
    -----------
    geometry : shapely.geometry.BaseGeometry
        Input geometry to regularize
    parallel_threshold : float
        Distance threshold for parallel line handling

    Returns:
    --------
    shapely.geometry.BaseGeometry
        Regularized geometry
    """
    if isinstance(geometry, Polygon):
        return regularize_single_polygon(
            polygon=geometry,
            parallel_threshold=parallel_threshold,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
            allow_circles=allow_circles,
            circle_threshold=circle_threshold,
            include_metadata=include_metadata,
        )

    else:
        # Return unmodified if not a polygon
        warnings.warn(
            f"Unsupported geometry type: {type(geometry)}. Returning original."
        )
        return geometry, None, None
