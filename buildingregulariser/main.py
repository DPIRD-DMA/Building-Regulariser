# import math
# import warnings
# from concurrent.futures import ProcessPoolExecutor
# from typing import Any, Callable, List, Optional, Tuple, Union, cast

# import geopandas as gpd
# import numpy as np
# import pyproj
# import shapely
# from pyproj import CRS
# from shapely.geometry import LinearRing, MultiPolygon, Polygon
# from shapely.geometry.base import BaseGeometry


# # RDP algorithm with improved naming
# def point_to_line_distance(
#     point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
# ) -> float:
#     """
#     Calculates the perpendicular distance from a point to a line defined by two points.

#     Parameters:
#     -----------
#     point : array-like
#         The point to calculate distance from
#     line_start : array-like
#         Starting point of the line
#     line_end : array-like
#         Ending point of the line

#     Returns:
#     --------
#     float
#         Distance from point to line
#     """
#     point, line_start, line_end = point[:2], line_start[:2], line_end[:2]  # ensure 2D
#     if line_start[0] == line_end[0]:
#         return np.abs(point[0] - line_start[0])

#     return np.divide(
#         np.linalg.norm(np.linalg.det([line_end - line_start, line_start - point])),
#         np.linalg.norm(line_end - line_start),
#     )


# def _ramer_douglas_peucker_recursive(
#     points: np.ndarray, epsilon: float, distance_func: Callable
# ) -> np.ndarray:
#     """
#     Recursive implementation of the Ramer-Douglas-Peucker algorithm.

#     Parameters:
#     -----------
#     points : numpy.ndarray
#         Array of points to simplify
#     epsilon : float
#         Simplification threshold
#     distance_func : function
#         Function to calculate point-to-line distance

#     Returns:
#     --------
#     numpy.ndarray
#         Simplified points
#     """
#     max_distance = 0.0
#     index_of_furthest = -1

#     # Find point with max distance from line between first and last point
#     for i in range(1, points.shape[0]):
#         distance = distance_func(points[i], points[0], points[-1])
#         if distance > max_distance:
#             index_of_furthest = i
#             max_distance = distance

#     # If max distance exceeds epsilon, recursively simplify
#     if max_distance > epsilon:
#         # Recursive call for first segment
#         first_segment = _ramer_douglas_peucker_recursive(
#             points[: index_of_furthest + 1], epsilon, distance_func
#         )
#         # Recursive call for second segment
#         second_segment = _ramer_douglas_peucker_recursive(
#             points[index_of_furthest:], epsilon, distance_func
#         )
#         # Combine simplified segments
#         return np.vstack((first_segment[:-1], second_segment))
#     else:
#         # If max distance is below epsilon, just use endpoints
#         return np.vstack((points[0], points[-1]))


# def _rdp_numpy_array(
#     points_list: List[np.ndarray], epsilon: float, distance_func: Callable
# ) -> List[List[float]]:
#     """
#     RDP simplification for list of points (converts to numpy array internally)
#     """
#     return _ramer_douglas_peucker_recursive(
#         np.array(points_list), epsilon, distance_func
#     ).tolist()


# def simplify_with_rdp(
#     points: Union[np.ndarray, List],
#     epsilon: float = 0,
#     distance_func: Callable = point_to_line_distance,
# ) -> Union[np.ndarray, List]:
#     """
#     Simplifies a given array of points using Ramer-Douglas-Peucker algorithm.

#     Parameters:
#     -----------
#     points : numpy.ndarray or list
#         Points to simplify
#     epsilon : float
#         Simplification threshold (higher = more simplification)
#     distance_func : function
#         Function to calculate point-to-line distance

#     Returns:
#     --------
#     numpy.ndarray or list
#         Simplified points
#     """
#     if "numpy" in str(type(points)):
#         return _ramer_douglas_peucker_recursive(points, epsilon, distance_func)
#     return _rdp_numpy_array(points, epsilon, distance_func)


# # Distance and angle calculation functions
# def calculate_distance(point_1: np.ndarray, point_2: np.ndarray) -> float:
#     """Calculate Euclidean distance between two points"""
#     return np.sqrt(np.sum(np.power((point_1 - point_2), 2)))


# def calculate_azimuth_angle(
#     start_point: Union[np.ndarray, Tuple[float, float]],
#     end_point: Union[np.ndarray, Tuple[float, float]],
# ) -> float:
#     """
#     Calculate azimuth angle of line between two points (in degrees)

#     Returns angle with respect to horizontal axis (x-axis)
#     """
#     x1, y1 = start_point
#     x2, y2 = end_point

#     if x1 < x2:
#         if y1 < y2:
#             angle = math.atan((y2 - y1) / (x2 - x1))
#             angle_degrees = angle * 180 / math.pi
#             return angle_degrees
#         elif y1 > y2:
#             angle = math.atan((y1 - y2) / (x2 - x1))
#             angle_degrees = angle * 180 / math.pi
#             return 90 + (90 - angle_degrees)
#         else:  # y1 == y2
#             return 0
#     elif x1 > x2:
#         if y1 < y2:
#             angle = math.atan((y2 - y1) / (x1 - x2))
#             angle_degrees = angle * 180 / math.pi
#             return 90 + (90 - angle_degrees)
#         elif y1 > y2:
#             angle = math.atan((y1 - y2) / (x1 - x2))
#             angle_degrees = angle * 180 / math.pi
#             return angle_degrees
#         else:  # y1 == y2
#             return 0
#     else:  # x1 == x2
#         return 90


# # Line operations with descriptive names
# def create_line_equation(
#     point1: Union[np.ndarray, Tuple[float, float]],
#     point2: Union[np.ndarray, Tuple[float, float]],
# ) -> Tuple[float, float, float]:
#     """
#     Create a line equation in the form Ax + By + C = 0

#     Parameters:
#     -----------
#     point1, point2 : array-like
#         Two points defining the line

#     Returns:
#     --------
#     tuple
#         Coefficients (A, B, C) where Ax + By + C = 0
#     """
#     A = point1[1] - point2[1]
#     B = point2[0] - point1[0]
#     C = point1[0] * point2[1] - point2[0] * point1[1]
#     return A, B, -C


# def calculate_line_intersection(
#     line1: Tuple[float, float, float], line2: Tuple[float, float, float]
# ) -> Union[Tuple[float, float], bool]:
#     """
#     Calculate intersection point of two lines

#     Parameters:
#     -----------
#     line1, line2 : tuple
#         Line coefficients (A, B, C) where Ax + By + C = 0

#     Returns:
#     --------
#     tuple or False
#         Coordinates of intersection point or False if lines are parallel
#     """
#     D = line1[0] * line2[1] - line1[1] * line2[0]
#     Dx = line1[2] * line2[1] - line1[1] * line2[2]
#     Dy = line1[0] * line2[2] - line1[2] * line2[0]
#     if D != 0:
#         x = Dx / D
#         y = Dy / D
#         return x, y
#     else:
#         return False


# def calculate_parallel_line_distance(
#     line1: Tuple[float, float, float], line2: Tuple[float, float, float]
# ) -> float:
#     """
#     Calculate the distance between two parallel lines

#     Parameters:
#     -----------
#     line1, line2 : tuple
#         Line coefficients (A, B, C) where Ax + By + C = 0

#     Returns:
#     --------
#     float
#         Distance between lines
#     """
#     A1, B1, C1 = line1
#     A2, B2, C2 = line2

#     # Normalize equations to the form: x + (B/A)y + (C/A) = 0
#     new_A1 = 1
#     new_B1 = B1 / A1
#     new_C1 = C1 / A1

#     new_A2 = 1
#     new_B2 = B2 / A2
#     new_C2 = C2 / A2

#     # Calculate distance using the formula for parallel lines
#     distance = (np.abs(new_C1 - new_C2)) / (np.sqrt(new_A2 * new_A2 + new_B2 * new_B2))
#     return distance


# def project_point_to_line(
#     point_x: float,
#     point_y: float,
#     line_x1: float,
#     line_y1: float,
#     line_x2: float,
#     line_y2: float,
# ) -> Tuple[float, float]:
#     """
#     Project a point onto a line

#     Parameters:
#     -----------
#     point_x, point_y : float
#         Coordinates of the point to project
#     line_x1, line_y1, line_x2, line_y2 : float
#         Coordinates of two points defining the line

#     Returns:
#     --------
#     tuple
#         Coordinates of the projected point
#     """
#     # Calculate the projected point coordinates
#     x = (
#         point_x * (line_x2 - line_x1) * (line_x2 - line_x1)
#         + point_y * (line_y2 - line_y1) * (line_x2 - line_x1)
#         + (line_x1 * line_y2 - line_x2 * line_y1) * (line_y2 - line_y1)
#     ) / (
#         (line_x2 - line_x1) * (line_x2 - line_x1)
#         + (line_y2 - line_y1) * (line_y2 - line_y1)
#     )

#     y = (
#         point_x * (line_x2 - line_x1) * (line_y2 - line_y1)
#         + point_y * (line_y2 - line_y1) * (line_y2 - line_y1)
#         + (line_x2 * line_y1 - line_x1 * line_y2) * (line_x2 - line_x1)
#     ) / (
#         (line_x2 - line_x1) * (line_x2 - line_x1)
#         + (line_y2 - line_y1) * (line_y2 - line_y1)
#     )

#     return (x, y)


# # Rotation functions
# def rotate_point_clockwise(
#     point: Union[np.ndarray, Tuple[float, float]],
#     center: Union[np.ndarray, Tuple[float, float]],
#     angle_degrees: float,
# ) -> Tuple[float, float]:
#     """
#     Rotate a point clockwise around a center point

#     Parameters:
#     -----------
#     point : array-like
#         Point to rotate
#     center : array-like
#         Center of rotation
#     angle_degrees : float
#         Rotation angle in degrees

#     Returns:
#     --------
#     tuple
#         Rotated point coordinates
#     """
#     x, y = point
#     center_x, center_y = center
#     angle_radians = math.radians(angle_degrees)

#     # Translate point to origin
#     translated_x = x - center_x
#     translated_y = y - center_y

#     # Rotate
#     rotated_x = translated_x * math.cos(angle_radians) + translated_y * math.sin(
#         angle_radians
#     )
#     rotated_y = translated_y * math.cos(angle_radians) - translated_x * math.sin(
#         angle_radians
#     )

#     # Translate back
#     final_x = rotated_x + center_x
#     final_y = rotated_y + center_y

#     return (final_x, final_y)


# def rotate_point_counterclockwise(
#     point: Union[np.ndarray, Tuple[float, float]],
#     center: Union[np.ndarray, Tuple[float, float]],
#     angle_degrees: float,
# ) -> Tuple[float, float]:
#     """
#     Rotate a point counter-clockwise around a center point

#     Parameters:
#     -----------
#     point : array-like
#         Point to rotate
#     center : array-like
#         Center of rotation
#     angle_degrees : float
#         Rotation angle in degrees

#     Returns:
#     --------
#     tuple
#         Rotated point coordinates
#     """
#     x, y = point
#     center_x, center_y = center
#     angle_radians = math.radians(angle_degrees)

#     # Translate point to origin
#     translated_x = x - center_x
#     translated_y = y - center_y

#     # Rotate
#     rotated_x = translated_x * math.cos(angle_radians) - translated_y * math.sin(
#         angle_radians
#     )
#     rotated_y = translated_x * math.sin(angle_radians) + translated_y * math.cos(
#         angle_radians
#     )

#     # Translate back
#     final_x = rotated_x + center_x
#     final_y = rotated_y + center_y

#     return (final_x, final_y)


# # Core regularization function (adapted from original)
# def regularize_coordinate_array(
#     coordinates: np.ndarray, epsilon: float = 6, parallel_threshold: float = 3
# ) -> np.ndarray:
#     """
#     Regularize polygon coordinates by aligning edges to be either parallel
#     or perpendicular to the longest edge

#     Parameters:
#     -----------
#     coordinates : numpy.ndarray
#         Array of coordinates for a polygon ring (shape: n x 2)
#     epsilon : float
#         Parameter for Ramer-Douglas-Peucker algorithm (higher = more simplification)
#     parallel_threshold : float
#         Distance threshold for considering parallel lines as needing connection

#     Returns:
#     --------
#     numpy.ndarray
#         Regularized coordinates array
#     """
#     # Ensure coordinates are closed
#     if not np.array_equal(coordinates[0], coordinates[-1]):
#         coordinates = np.vstack([coordinates, coordinates[0]])

#     # Simplify using RDP algorithm
#     simplified_coordinates = simplify_with_rdp(coordinates, epsilon=epsilon)

#     # Get lengths and angles of each segment
#     edge_lengths: List[float] = []
#     azimuth_angles: List[float] = []
#     edge_indices: List[List[int]] = []

#     for i in range(len(simplified_coordinates) - 1):
#         current_point = simplified_coordinates[i]
#         next_point = simplified_coordinates[i + 1]

#         edge_length = calculate_distance(current_point, next_point)
#         azimuth = calculate_azimuth_angle(current_point, next_point)

#         edge_lengths.append(edge_length)
#         azimuth_angles.append(azimuth)
#         edge_indices.append([i, i + 1])

#     # Find longest edge and use its direction as main direction
#     longest_edge_index = np.argmax(edge_lengths)
#     main_direction = azimuth_angles[longest_edge_index]

#     # Correct directions by rotating to be parallel or perpendicular to main direction
#     oriented_edges: List[List[Union[Tuple[float, float], np.ndarray]]] = []
#     edge_orientations: List[int] = []  # 0=parallel, 1=perpendicular

#     for i, (azimuth, (start_idx, end_idx)) in enumerate(
#         zip(azimuth_angles, edge_indices)
#     ):
#         if i == longest_edge_index:
#             # Keep longest edge as is
#             oriented_edges.append(
#                 [simplified_coordinates[start_idx], simplified_coordinates[end_idx]]
#             )
#             edge_orientations.append(0)  # Parallel to main direction
#         else:
#             # Determine rotation angle to align with main direction
#             rotation_angle = main_direction - azimuth

#             if np.abs(rotation_angle) < 45:  # 180/4 = 45
#                 # Make parallel to main direction
#                 edge_orientations.append(0)
#             elif np.abs(rotation_angle) >= 45:
#                 # Make perpendicular to main direction
#                 rotation_angle = rotation_angle + 90
#                 edge_orientations.append(1)

#             # Perform rotation
#             start_point = simplified_coordinates[start_idx]
#             end_point = simplified_coordinates[end_idx]
#             midpoint = (start_point + end_point) / 2

#             if rotation_angle > 0:
#                 rotated_start = rotate_point_counterclockwise(
#                     start_point, midpoint, np.abs(rotation_angle)
#                 )
#                 rotated_end = rotate_point_counterclockwise(
#                     end_point, midpoint, np.abs(rotation_angle)
#                 )
#             elif rotation_angle < 0:
#                 rotated_start = rotate_point_clockwise(
#                     start_point, midpoint, np.abs(rotation_angle)
#                 )
#                 rotated_end = rotate_point_clockwise(
#                     end_point, midpoint, np.abs(rotation_angle)
#                 )
#             else:
#                 rotated_start = start_point
#                 rotated_end = end_point

#             oriented_edges.append([rotated_start, rotated_end])

#     oriented_edges_array = np.array(oriented_edges, dtype=object)

#     # Fix adjacent edges (intersection for perpendicular, connection for parallel)
#     regularized_points: List[Union[Tuple[float, float], np.ndarray]] = []
#     regularized_points.append(oriented_edges_array[0][0])

#     for i in range(len(oriented_edges_array) - 1):
#         current_index = i
#         next_index = i + 1 if i < len(oriented_edges_array) - 1 else 0

#         current_edge_start = oriented_edges_array[current_index][0]
#         current_edge_end = oriented_edges_array[current_index][1]
#         next_edge_start = oriented_edges_array[next_index][0]
#         next_edge_end = oriented_edges_array[next_index][1]

#         current_orientation = edge_orientations[current_index]
#         next_orientation = edge_orientations[next_index]

#         if current_orientation != next_orientation:
#             # For perpendicular edges, find intersection
#             line1 = create_line_equation(current_edge_start, current_edge_end)
#             line2 = create_line_equation(next_edge_start, next_edge_end)

#             intersection_point = calculate_line_intersection(line1, line2)
#             if intersection_point:
#                 regularized_points.append(intersection_point)
#             else:
#                 # If lines are parallel (shouldn't happen with perpendicular check)
#                 # add the end point of current edge
#                 regularized_points.append(current_edge_end)

#         elif current_orientation == next_orientation:
#             # For parallel edges, add connection based on distance
#             line1 = create_line_equation(current_edge_start, current_edge_end)
#             line2 = create_line_equation(next_edge_start, next_edge_end)
#             line_distance = calculate_parallel_line_distance(line1, line2)

#             if line_distance < parallel_threshold:
#                 # Shift next edge to align with current edge
#                 projected_point = project_point_to_line(
#                     next_edge_start[0],
#                     next_edge_start[1],
#                     current_edge_start[0],
#                     current_edge_start[1],
#                     current_edge_end[0],
#                     current_edge_end[1],
#                 )
#                 regularized_points.append(projected_point)

#                 # Update next edge starting point
#                 oriented_edges_array[next_index][0] = projected_point
#                 oriented_edges_array[next_index][1] = project_point_to_line(
#                     next_edge_end[0],
#                     next_edge_end[1],
#                     current_edge_start[0],
#                     current_edge_start[1],
#                     current_edge_end[0],
#                     current_edge_end[1],
#                 )
#             else:
#                 # Add connecting segment between edges
#                 midpoint = (current_edge_end + next_edge_start) / 2
#                 connecting_point1 = project_point_to_line(
#                     midpoint[0],
#                     midpoint[1],
#                     current_edge_start[0],
#                     current_edge_start[1],
#                     current_edge_end[0],
#                     current_edge_end[1],
#                 )
#                 connecting_point2 = project_point_to_line(
#                     midpoint[0],
#                     midpoint[1],
#                     next_edge_start[0],
#                     next_edge_start[1],
#                     next_edge_end[0],
#                     next_edge_end[1],
#                 )
#                 regularized_points.append(connecting_point1)
#                 regularized_points.append(connecting_point2)

#     # Close the polygon
#     if len(regularized_points) > 0 and not np.array_equal(
#         regularized_points[0], regularized_points[-1]
#     ):
#         regularized_points.append(regularized_points[0])

#     return np.array(regularized_points, dtype=object)


# def regularize_single_polygon(
#     polygon: Polygon, epsilon: float = 6, parallel_threshold: float = 3
# ) -> Polygon:
#     """
#     Regularize a Shapely polygon by aligning edges to principal directions

#     Parameters:
#     -----------
#     polygon : shapely.geometry.Polygon
#         Input polygon to regularize
#     epsilon : float
#         Parameter for Ramer-Douglas-Peucker algorithm (higher = more simplification)
#     parallel_threshold : float
#         Distance threshold for parallel line handling

#     Returns:
#     --------
#     shapely.geometry.Polygon
#         Regularized polygon
#     """
#     # Handle exterior ring
#     exterior_coordinates = np.array(polygon.exterior.coords)
#     regularized_exterior = regularize_coordinate_array(
#         exterior_coordinates, epsilon=epsilon, parallel_threshold=parallel_threshold
#     )

#     # Handle interior rings (holes)
#     regularized_interiors: List[np.ndarray] = []
#     for interior in polygon.interiors:
#         interior_coordinates = np.array(interior.coords)
#         regularized_interior = regularize_coordinate_array(
#             interior_coordinates, epsilon=epsilon, parallel_threshold=parallel_threshold
#         )
#         regularized_interiors.append(regularized_interior)

#     # Create new polygon
#     try:
#         # Convert coordinates to LinearRings
#         exterior_ring = LinearRing(regularized_exterior)
#         interior_rings = [LinearRing(r) for r in regularized_interiors]

#         # Create regularized polygon
#         regularized_polygon = Polygon(exterior_ring, interior_rings)
#         return regularized_polygon
#     except Exception as e:
#         # If there's an error creating the polygon, return the original
#         warnings.warn(f"Error creating regularized polygon: {e}. Returning original.")
#         return polygon


# def process_geometry(
#     geometry: BaseGeometry, epsilon: float, parallel_threshold: float
# ) -> BaseGeometry:
#     """
#     Process a single geometry, handling different geometry types

#     Parameters:
#     -----------
#     geometry : shapely.geometry.BaseGeometry
#         Input geometry to regularize
#     epsilon : float
#         Parameter for Ramer-Douglas-Peucker algorithm
#     parallel_threshold : float
#         Distance threshold for parallel line handling

#     Returns:
#     --------
#     shapely.geometry.BaseGeometry
#         Regularized geometry
#     """
#     if isinstance(geometry, Polygon):
#         return regularize_single_polygon(geometry, epsilon, parallel_threshold)
#     elif isinstance(geometry, MultiPolygon):
#         regularized_parts = [
#             regularize_single_polygon(part, epsilon, parallel_threshold)
#             for part in geometry.geoms
#         ]
#         return MultiPolygon(regularized_parts)
#     else:
#         # Return unmodified if not a polygon
#         warnings.warn(
#             f"Unsupported geometry type: {type(geometry)}. Returning original."
#         )
#         return geometry


# def regularize_geodataframe(
#     geodataframe: gpd.GeoDataFrame,
#     epsilon: float = 6,
#     parallel_threshold: float = 3,
#     target_crs: Optional[Union[str, pyproj.CRS]] = None,
#     check_projection: bool = True,
#     n_jobs: int = 1,
# ) -> gpd.GeoDataFrame:
#     """
#     Regularize polygons in a GeoDataFrame by aligning edges to principal directions

#     Parameters:
#     -----------
#     geodataframe : geopandas.GeoDataFrame
#         Input GeoDataFrame with polygon geometries
#     epsilon : float
#         Parameter for Ramer-Douglas-Peucker algorithm (higher = more simplification)
#     parallel_threshold : float
#         Distance threshold for handling parallel lines
#     target_crs : str or pyproj.CRS, optional
#         Target CRS for reprojection. If None, uses the input CRS.
#     check_projection : bool
#         If True, checks if the data is in a projected CRS and warns if not
#     n_jobs : int
#         Number of parallel processes to use (future implementation)

#     Returns:
#     --------
#     geopandas.GeoDataFrame
#         GeoDataFrame with regularized polygons
#     """
#     # Make a copy to avoid modifying the original
#     result_geodataframe = geodataframe.copy()

#     # Check if input has CRS
#     if geodataframe.crs is None:
#         warnings.warn("Input GeoDataFrame has no CRS defined.")

#     # Handle CRS reprojection and check
#     original_crs = geodataframe.crs

#     if target_crs is not None:
#         # Reproject to specified CRS
#         result_geodataframe = result_geodataframe.to_crs(target_crs)
#     elif check_projection and geodataframe.crs is not None:
#         # Check if current CRS is projected
#         crs = CRS.from_user_input(geodataframe.crs)
#         if not crs.is_projected:
#             warnings.warn(
#                 "Input GeoDataFrame is in a geographic CRS (not projected). "
#                 "This may affect angle calculations. Consider setting target_crs."
#             )

#     # Process geometries
#     if n_jobs == 1:
#         # Single process implementation
#         result_geodataframe["geometry"] = result_geodataframe.geometry.apply(
#             lambda geom: process_geometry(geom, epsilon, parallel_threshold)
#         )
#     else:
#         # Placeholder for future multiprocessing implementation
#         warnings.warn("Multiprocessing not yet implemented. Using single process.")
#         result_geodataframe["geometry"] = result_geodataframe.geometry.apply(
#             lambda geom: process_geometry(geom, epsilon, parallel_threshold)
#         )

#     # Reproject back to original CRS if necessary
#     if target_crs is not None and original_crs is not None:
#         result_geodataframe = result_geodataframe.to_crs(original_crs)

#     return result_geodataframe
