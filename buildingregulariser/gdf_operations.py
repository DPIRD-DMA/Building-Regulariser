"""
GeoDataFrame operations for polygon regularization.

Functions for applying regularization to GeoDataFrame objects with polygon geometries.
"""

import warnings
from functools import partial  # Helpful for passing fixed arguments
from multiprocessing import Pool
from typing import Optional, Union

import geopandas as gpd
import pyproj
from pyproj import CRS

from .regularization import process_geometry


def cleanup_geometry(result_geodataframe: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result_geodataframe = result_geodataframe[~result_geodataframe.geometry.is_empty]

    result_geodataframe["geometry"] = result_geodataframe.geometry.buffer(
        0.001, cap_style="square", join_style="mitre"
    )
    result_geodataframe["geometry"] = result_geodataframe.geometry.buffer(
        -0.002, cap_style="square", join_style="mitre"
    )
    result_geodataframe["geometry"] = result_geodataframe.geometry.buffer(
        0.001, cap_style="square", join_style="mitre"
    )
    result_geodataframe = result_geodataframe[~result_geodataframe.geometry.is_empty]
    return result_geodataframe


def regularize_geodataframe(
    geodataframe: gpd.GeoDataFrame,
    parallel_threshold: float = 3,
    target_crs: Optional[Union[str, pyproj.CRS]] = None,
    check_projection: bool = True,
    simplify: bool = True,
    simplify_tolerance: float = 0.5,
    processes: int = 1,
) -> gpd.GeoDataFrame:
    """
    Regularize polygons in a GeoDataFrame by aligning edges to principal directions

    Parameters:
    -----------
    geodataframe : geopandas.GeoDataFrame
        Input GeoDataFrame with polygon geometries
    epsilon : float
        Parameter for Ramer-Douglas-Peucker algorithm (higher = more simplification)
    parallel_threshold : float
        Distance threshold for handling parallel lines
    target_crs : str or pyproj.CRS, optional
        Target CRS for reprojection. If None, uses the input CRS.
    check_projection : bool
        If True, checks if the data is in a projected CRS and warns if not
    simplify : bool
        If True, applies simplification to the geometry
    simplify_tolerance : float
        Tolerance for simplification. Default is 0.5.
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with regularized polygons
    """
    # Make a copy to avoid modifying the original
    result_geodataframe = geodataframe.copy()

    # Check if input has CRS
    if geodataframe.crs is None:
        warnings.warn("Input GeoDataFrame has no CRS defined.")

    # Handle CRS reprojection and check
    original_crs = geodataframe.crs

    if target_crs is not None:
        # Reproject to specified CRS
        result_geodataframe = result_geodataframe.to_crs(target_crs)
    elif check_projection and geodataframe.crs is not None:
        # Check if current CRS is projected
        crs = CRS.from_user_input(geodataframe.crs)
        if not crs.is_projected:
            warnings.warn(
                "Input GeoDataFrame is in a geographic CRS (not projected). "
                "This may affect angle calculations. Consider setting target_crs."
            )
    if simplify:
        result_geodataframe.geometry = result_geodataframe.simplify(
            tolerance=simplify_tolerance, preserve_topology=True
        )
    if processes > 1:
        # Parallel processing
        with Pool(processes) as pool:
            # Use partial to fix the arguments for the process_geometry function
            process_func = partial(
                process_geometry, parallel_threshold=parallel_threshold
            )
            result_geodataframe["geometry"] = pool.map(
                process_func, result_geodataframe.geometry.tolist()
            )
    else:
        # Single process implementation
        result_geodataframe["geometry"] = result_geodataframe.geometry.apply(
            lambda geom: process_geometry(geom, parallel_threshold)  # type: ignore
        )  # type: ignore
    # Cleanup geometry
    result_geodataframe = cleanup_geometry(result_geodataframe)

    # Reproject back to original CRS if necessary
    if target_crs is not None and original_crs is not None:
        result_geodataframe = result_geodataframe.to_crs(original_crs)

    return result_geodataframe
