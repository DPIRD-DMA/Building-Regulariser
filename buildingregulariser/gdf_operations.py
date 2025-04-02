import warnings
from functools import partial
from multiprocessing import Pool
from typing import Optional, Union

import geopandas as gpd
import pyproj
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from .regularization import process_geometry


def cleanup_geometry(
    result_geodataframe: gpd.GeoDataFrame, simplify_tolerance: float
) -> gpd.GeoDataFrame:
    """
    Cleans up geometries in a GeoDataFrame.

    Removes empty geometries, attempts to remove small slivers using buffer
    operations, and simplifies geometries to remove redundant vertices.

    Parameters:
    -----------
    result_geodataframe : geopandas.GeoDataFrame
        GeoDataFrame with geometries to clean.
    simplify_tolerance : float
        Tolerance used for simplification and determining buffer size
        for sliver removal.

    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with cleaned geometries.
    """
    # Remove empty geometries before buffering
    result_geodataframe = result_geodataframe[~result_geodataframe.geometry.is_empty]
    if result_geodataframe.empty:
        return result_geodataframe  # Return early if GDF is empty

    # Define buffer size based on simplify tolerance
    buffer_size = simplify_tolerance / 50

    # Attempt to remove small slivers using a sequence of buffer operations
    # Positive buffer -> negative buffer -> positive buffer
    result_geodataframe["geometry"] = result_geodataframe.geometry.buffer(
        buffer_size, cap_style="square", join_style="mitre"
    )
    result_geodataframe["geometry"] = result_geodataframe.geometry.buffer(
        buffer_size * -2, cap_style="square", join_style="mitre"
    )
    result_geodataframe["geometry"] = result_geodataframe.geometry.buffer(
        buffer_size, cap_style="square", join_style="mitre"
    )

    # Remove any geometries that became empty after buffering
    result_geodataframe = result_geodataframe[~result_geodataframe.geometry.is_empty]
    if result_geodataframe.empty:
        return result_geodataframe  # Return early if GDF is empty

    # Simplify to remove collinear vertices introduced by buffering/regularization
    # Use a small tolerance related to the buffer size
    result_geodataframe["geometry"] = result_geodataframe.geometry.simplify(
        tolerance=buffer_size, preserve_topology=True
    )
    # Final check for empty geometries after simplification
    result_geodataframe = result_geodataframe[~result_geodataframe.geometry.is_empty]

    return result_geodataframe


def process_geometry_wrapper(
    geom: BaseGeometry,
    parallel_threshold: float,
    allow_45_degree: bool,
    allow_circles: bool,
    circle_threshold: float,
) -> Optional[BaseGeometry]:
    """
    Wrapper function for `process_geometry` for use with multiprocessing.

    Includes basic error handling to prevent single geometry failures from
    stopping the entire pool.

    Parameters:
    -----------
    geom : shapely.geometry.BaseGeometry
        The input geometry to process.
    parallel_threshold : float
        Distance threshold for handling parallel lines in regularization.
    allow_45_degree : bool
        Flag to allow 45-degree angles during regularization.
    allow_circles : bool
        Flag to allow detection and replacement of polygons with circles.
    circle_threshold : float
        IOU threshold to consider a polygon a circle.

    Returns:
    --------
    Optional[shapely.geometry.BaseGeometry]
        The processed geometry, or None if an error occurred during processing.
    """
    try:
        return process_geometry(
            geom, parallel_threshold, allow_45_degree, allow_circles, circle_threshold
        )
    except Exception as e:
        # Warn about the error but allow other processes to continue
        warnings.warn(
            f"Error processing geometry in parallel worker: {e}. Skipping geometry."
        )
        return None  # Return None to indicate failure for this geometry


def regularize_geodataframe(
    geodataframe: gpd.GeoDataFrame,
    parallel_threshold: float = 1.0,  # Default value was 1 in docstring but not in signature
    target_crs: Optional[Union[str, pyproj.CRS]] = None,
    simplify: bool = True,
    simplify_tolerance: float = 0.5,
    allow_45_degree: bool = True,
    allow_circles: bool = True,
    circle_threshold: float = 0.9,
    num_cores: int = 1,
) -> gpd.GeoDataFrame:
    """
    Regularizes polygon geometries in a GeoDataFrame by aligning edges.

    Aligns edges to be parallel or perpendicular (optionally also 45 degrees)
    to their main direction. Handles reprojection, initial simplification,
    regularization, geometry cleanup, and parallel processing.

    Parameters:
    -----------
    geodataframe : geopandas.GeoDataFrame
        Input GeoDataFrame with polygon or multipolygon geometries.
    parallel_threshold : float, optional
        Distance threshold for merging nearly parallel adjacent edges during
        regularization. Defaults to 1.0.
    target_crs : str or pyproj.CRS, optional
        Target Coordinate Reference System for processing. If None, uses the
        input GeoDataFrame's CRS. Processing is more reliable in a projected CRS.
        Defaults to None.
    simplify : bool, optional
        If True, applies initial simplification to the geometry before
        regularization. Defaults to True.
    simplify_tolerance : float, optional
        Tolerance for the initial simplification step (if `simplify` is True).
        Also used for geometry cleanup steps. Defaults to 0.5.
    allow_45_degree : bool, optional
        If True, allows edges to be oriented at 45-degree angles relative
        to the main direction during regularization. Defaults to True.
    allow_circles : bool, optional
        If True, attempts to detect polygons that are nearly circular and
        replaces them with perfect circles. Defaults to True.
    circle_threshold : float, optional
        Intersection over Union (IoU) threshold used for circle detection
        (if `allow_circles` is True). Value between 0 and 1. Defaults to 0.9.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. If 1, processing
        is done sequentially. Defaults to 1.

    Returns:
    --------
    geopandas.GeoDataFrame
        A new GeoDataFrame with regularized polygon geometries. Original
        attributes are preserved. Geometries that failed processing might be
        dropped.
    """
    # Make a copy to avoid modifying the original GeoDataFrame
    result_geodataframe = geodataframe.copy()

    # Check if input has CRS defined, warn if not
    if result_geodataframe.crs is None:
        warnings.warn(
            "Input GeoDataFrame has no CRS defined. Assuming planar coordinates."
        )

    # Store original CRS for potential reprojection back at the end
    original_crs = result_geodataframe.crs

    # Reproject to target CRS if specified
    if target_crs is not None:
        result_geodataframe = result_geodataframe.to_crs(target_crs)

    # Check if the CRS used for processing is projected, warn if not
    # Use the potentially reprojected CRS
    current_crs = result_geodataframe.crs
    if current_crs:  # Check if CRS exists before trying to use it
        crs_obj = CRS.from_user_input(current_crs)
        if not crs_obj.is_projected:
            warnings.warn(
                f"GeoDataFrame is in a geographic CRS ('{current_crs.name}') during processing. "
                "Angle and distance calculations may be inaccurate. Consider setting "
                "`target_crs` to a suitable projected CRS."
            )

    # Apply initial simplification if requested
    if simplify:
        result_geodataframe.geometry = result_geodataframe.simplify(
            tolerance=simplify_tolerance, preserve_topology=True
        )
        # Remove geometries that might become invalid after simplification
        result_geodataframe = result_geodataframe[result_geodataframe.geometry.is_valid]
        result_geodataframe = result_geodataframe[
            ~result_geodataframe.geometry.is_empty
        ]
        if result_geodataframe.empty:
            warnings.warn("GeoDataFrame became empty after initial simplification.")
            return result_geodataframe  # Return early if empty

    # Create the partial function for processing geometries
    processing_func = partial(
        process_geometry_wrapper,  # Use the safe wrapper
        parallel_threshold=parallel_threshold,
        allow_45_degree=allow_45_degree,
        allow_circles=allow_circles,
        circle_threshold=circle_threshold,
    )

    # Apply the regularization function, using parallel processing if num_cores > 1
    if (
        num_cores > 1 and len(result_geodataframe) > num_cores
    ):  # Basic check if parallelization is useful
        with Pool(processes=num_cores) as pool:
            # Adjust chunksize based on typical geometry complexity and number of geometries
            chunk_size = max(1, min(100, len(result_geodataframe) // (num_cores * 2)))
            results = list(
                pool.imap_unordered(
                    processing_func, result_geodataframe.geometry, chunksize=chunk_size
                )
            )
        result_geodataframe["geometry"] = results
        # Filter out None results from processing errors
        result_geodataframe = result_geodataframe[result_geodataframe.geometry.notna()]

    else:  # Single core processing
        if num_cores > 1:
            warnings.warn(
                f"num_cores > 1 but GeoDataFrame size ({len(result_geodataframe)}) is small, using single core."
            )
        print("Starting regularization using single core...")
        # Use apply directly (casting necessary for type checker)
        result_geodataframe["geometry"] = result_geodataframe.geometry.apply(
            lambda geom: processing_func(geom)  # type: ignore
        )
        # Filter out None results from processing errors
        result_geodataframe = result_geodataframe[result_geodataframe.geometry.notna()]
        print("Single core processing finished.")

    if result_geodataframe.empty:
        warnings.warn("GeoDataFrame became empty after regularization processing.")
        return result_geodataframe  # Return early if empty

    # Clean up the resulting geometries (remove slivers, simplify)
    result_geodataframe = cleanup_geometry(
        result_geodataframe=result_geodataframe, simplify_tolerance=simplify_tolerance
    )

    if result_geodataframe.empty:
        warnings.warn("GeoDataFrame became empty after geometry cleanup.")
        return result_geodataframe  # Return early if empty

    # Reproject back to the original CRS if it was changed
    if target_crs is not None and original_crs is not None:
        # Check if CRS are actually different before reprojecting
        if not CRS.from_user_input(result_geodataframe.crs).equals(
            CRS.from_user_input(original_crs)
        ):
            result_geodataframe = result_geodataframe.to_crs(original_crs)

    return result_geodataframe
