from functools import partial
from multiprocessing import Pool
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import pyproj

from .chunk_processing import (
    get_chunk_size,
    process_geometry_wrapper,
    split_gdf,
)


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
    num_cores: int = 1,
    include_metadata: bool = False,
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
    diagonal_threshold_reduction : float, optional
        Reduction factor in degrees to reduce the likelihood of diagonal
        edges being created. larger values reduce the likelihood of diagonal edges.
        Defaults to 15.
    allow_circles : bool, optional
        If True, attempts to detect polygons that are nearly circular and
        replaces them with perfect circles. Defaults to True.
    circle_threshold : float, optional
        Intersection over Union (IoU) threshold used for circle detection
        (if `allow_circles` is True). Value between 0 and 1. Defaults to 0.9.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. If 1, processing
        is done sequentially. Defaults to 1.
    include_metadata : bool, optional
        If True, includes metadata about the regularization process in the
        output GeoDataFrame. Defaults to False.

    Returns:
    --------
    geopandas.GeoDataFrame
        A new GeoDataFrame with regularized polygon geometries. Original
        attributes are preserved. Geometries that failed processing might be
        dropped.
    """
    # Make a copy to avoid modifying the original GeoDataFrame
    result_geodataframe = geodataframe.copy()
    # explode the geometries to process them individually
    result_geodataframe = result_geodataframe.explode(ignore_index=True)
    # split gdf into chunks for parallel processing

    if num_cores == 1:
        result_geodataframe = process_geometry_wrapper(
            result_geodataframe=result_geodataframe,
            target_crs=target_crs,
            simplify=simplify,
            simplify_tolerance=simplify_tolerance,
            parallel_threshold=parallel_threshold,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
            allow_circles=allow_circles,
            circle_threshold=circle_threshold,
            include_metadata=include_metadata,
        )
    else:
        chunk_size = get_chunk_size(
            item_count=len(result_geodataframe), num_cores=num_cores
        )
        gdf_chunks = split_gdf(result_geodataframe, chunk_size=chunk_size)
        with Pool(processes=num_cores) as pool:
            # Use partial to pass additional arguments to the worker function
            process_geometry_partial = partial(
                process_geometry_wrapper,
                target_crs=target_crs,
                simplify=simplify,
                simplify_tolerance=simplify_tolerance,
                parallel_threshold=parallel_threshold,
                allow_45_degree=allow_45_degree,
                diagonal_threshold_reduction=diagonal_threshold_reduction,
                allow_circles=allow_circles,
                circle_threshold=circle_threshold,
                include_metadata=include_metadata,
            )
            # Process each chunk in parallel
            processed_chunks = pool.map(process_geometry_partial, gdf_chunks)

        result_geodataframe = gpd.GeoDataFrame(
            pd.concat(processed_chunks, ignore_index=True), crs=result_geodataframe.crs
        )
    return result_geodataframe
