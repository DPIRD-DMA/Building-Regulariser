"""
Polygon Regularization Package

A package for regularizing polygons by aligning edges to principal directions.
"""

from .__version__ import __version__
from .gdf_operations import regularize_geodataframe
from .geometry_utils import calculate_azimuth_angle, calculate_distance
from .line_operations import (
    calculate_line_intersection,
    calculate_parallel_line_distance,
    create_line_equation,
    project_point_to_line,
)
from .regularization import (
    process_geometry,
    regularize_coordinate_array,
    regularize_single_polygon,
)
from .rotation import rotate_point_clockwise, rotate_point_counterclockwise
from .simplification import simplify_with_rdp

# Package-wide exports
__all__ = [
    "simplify_with_rdp",
    "calculate_distance",
    "calculate_azimuth_angle",
    "create_line_equation",
    "calculate_line_intersection",
    "calculate_parallel_line_distance",
    "project_point_to_line",
    "rotate_point_clockwise",
    "rotate_point_counterclockwise",
    "regularize_coordinate_array",
    "regularize_single_polygon",
    "process_geometry",
    "regularize_geodataframe",
    "__version__",
]
