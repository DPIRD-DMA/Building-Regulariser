from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import pyproj

from .neighbor_alignment import align_with_neighbor_polygons
from .regularization import regularize_single_polygon


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
    # Filter out None results from processing errors
    result_geodataframe = result_geodataframe[~result_geodataframe.geometry.is_empty]
    result_geodataframe = result_geodataframe[result_geodataframe.geometry.notna()]

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


def regularize_geodataframe(
    geodataframe: gpd.GeoDataFrame,
    parallel_threshold: float = 1.0,
    target_crs: Optional[Union[str, pyproj.CRS]] = None,
    simplify: bool = True,
    simplify_tolerance: float = 0.5,
    allow_45_degree: bool = True,
    diagonal_threshold_reduction: float = 15,
    allow_circles: bool = True,
    circle_threshold: float = 0.9,
    num_cores: int = 0,
    include_metadata: bool = False,
    neighbor_alignment: bool = False,
    neighbor_search_distance: float = 100.0,
    neighbor_max_rotation: float = 10,
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
        regularization. Specified in the same units as the input GeoDataFrame's CRS. Defaults to 1.0.
    target_crs : str or pyproj.CRS, optional
        CRS to reproject the input GeoDataFrame to before regularization.
        If None, no reprojection is performed. Defaults to None.
    simplify : bool, optional
        If True, applies initial simplification to the geometry before
        regularization. Defaults to True.
    simplify_tolerance : float, optional
        Tolerance for the initial simplification step (if `simplify` is True).
        Also used for geometry cleanup steps. Specified in the same units as the input GeoDataFrame's CRS. Defaults to 0.5.
    allow_45_degree : bool, optional
        If True, allows edges to be oriented at 45-degree angles relative
        to the main direction during regularization. Defaults to True.
    diagonal_threshold_reduction : float, optional
        Reduction factor in degrees to reduce the likelihood of diagonal
        edges being created. larger values reduce the likelihood of diagonal edges. Possible values are 0 - 22.5 degrees.
        Defaults to 15 degrees.
    allow_circles : bool, optional
        If True, attempts to detect polygons that are nearly circular and
        replaces them with perfect circles. Defaults to True.
    circle_threshold : float, optional
        Intersection over Union (IoU) threshold used for circle detection
        (if `allow_circles` is True). Value between 0 and 1. Defaults to 0.9.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. If 1, processing
        is done sequentially. Defaults to 0 (all available cores).
    include_metadata : bool, optional
        If True, includes metadata about the regularization process in the
        output GeoDataFrame. Defaults to False.
    neighbor_alignment : bool, optional
        If True, aligns the polygons with their neighbors after regularization.
        Defaults to False.
    neighbor_search_distance : float, optional
        Search radius used to identify neighboring polygons for alignment (if `align_with_neighbors` is True).
        Specified in the same units as the input GeoDataFrame's CRS. Defaults to 100.0.
    neighbor_max_rotation : float, optional
        Direction threshold for aligning with neighbors (if
        `align_with_neighbors` is True). Defaults to 10 degrees.

    Returns:
    --------
    geopandas.GeoDataFrame
        A new GeoDataFrame with regularized polygon geometries. Original
        attributes are preserved. Geometries that failed processing might be
        dropped.
    """
    # Make a copy to avoid modifying the original GeoDataFrame
    result_geodataframe = geodataframe.copy()
    # Explode the geometries to process them individually
    result_geodataframe = result_geodataframe.explode(ignore_index=True)

    if target_crs is not None:
        # Reproject to the target CRS if specified
        result_geodataframe = result_geodataframe.to_crs(target_crs)
    # Split gdf into chunks for parallel processing
    # Determine number of jobs
    if num_cores <= 0:
        num_cores = cpu_count()

    partial_regularize_single_polygon = partial(
        regularize_single_polygon,
        parallel_threshold=parallel_threshold,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
        allow_circles=allow_circles,
        circle_threshold=circle_threshold,
        include_metadata=include_metadata,
        simplify=simplify,
        simplify_tolerance=simplify_tolerance,
    )

    # Sequential processing
    if num_cores == 1:
        processed_data = [
            partial_regularize_single_polygon(geometry)
            for geometry in result_geodataframe["geometry"]
        ]
    else:
        with Pool(num_cores) as p:
            processed_data = p.map(
                partial_regularize_single_polygon,
                result_geodataframe["geometry"],
            )

    results_df = pd.DataFrame(processed_data)
    result_geodataframe["geometry"] = results_df["geometry"]
    result_geodataframe["iou"] = results_df["iou"]
    result_geodataframe["main_direction"] = results_df["main_direction"]

    # Clean up the resulting geometries (remove slivers)
    result_geodataframe = cleanup_geometry(
        result_geodataframe=result_geodataframe, simplify_tolerance=simplify_tolerance
    )

    # Return result_geodataframe
    if neighbor_alignment:
        result_geodataframe = align_with_neighbor_polygons(
            gdf=result_geodataframe,
            buffer_size=neighbor_search_distance,
            max_rotation=neighbor_max_rotation,
            include_metadata=include_metadata,
            num_cores=num_cores,
        )

    if not include_metadata:
        # Extract metadata columns from the results DataFrame
        try_to_drop_cols = [
            "iou",
            "main_direction",
            "perimeter",
            "aligned_direction",
        ]
        for col in try_to_drop_cols:
            if col in result_geodataframe.columns:
                result_geodataframe.drop(columns=col, inplace=True)

    return result_geodataframe
