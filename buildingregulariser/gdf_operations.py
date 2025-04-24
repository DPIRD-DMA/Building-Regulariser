from functools import partial
from multiprocessing import Pool, cpu_count

import geopandas as gpd
import pandas as pd

from .chunk_processing import get_chunk_size, process_geometry_wrapper, split_gdf
from .neighbor_alignment import align_with_neighbor_polygons


def regularize_geodataframe(
    geodataframe: gpd.GeoDataFrame,
    parallel_threshold: float = 1.0,
    simplify: bool = True,
    simplify_tolerance: float = 0.5,
    allow_45_degree: bool = True,
    diagonal_threshold_reduction: float = 15,
    allow_circles: bool = True,
    circle_threshold: float = 0.9,
    num_cores: int = 0,
    include_metadata: bool = False,
    align_with_neighbors: bool = False,
    neighbors_buffer_size: float = 350.0,
    neighbors_min_count: int = 3,
    neighbors_direction_threshold: float = 10,
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
    align_with_neighbors : bool, optional
        If True, aligns the polygons with their neighbors after regularization.
        Defaults to False.
    neighbors_buffer_size : float, optional
        Search radius used to identify neighboring polygons for alignment (if `align_with_neighbors` is True).
        Specified in the same units as the input GeoDataFrame's CRS. Defaults to 350.0.
    neighbors_min_count : int, optional
        Minimum number of neighbors required for alignment (if
        `align_with_neighbors` is True). Defaults to 3.
    neighbors_direction_threshold : float, optional
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
    # Split gdf into chunks for parallel processing
    # Determine number of jobs
    if num_cores <= 0:
        num_cores = cpu_count()

    if num_cores == 1:
        result_geodataframe = process_geometry_wrapper(
            result_geodataframe=result_geodataframe,
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

    # Return result_geodataframe
    if align_with_neighbors:
        result_geodataframe = align_with_neighbor_polygons(
            gdf=result_geodataframe,
            buffer_size=neighbors_buffer_size,
            min_count=neighbors_min_count,
            direction_threshold=neighbors_direction_threshold,
            include_metadata=include_metadata,
            num_cores=num_cores,
        )

    return result_geodataframe
