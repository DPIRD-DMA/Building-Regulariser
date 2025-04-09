import warnings
from typing import Optional, Union

import geopandas as gpd
import pyproj
from pyproj import CRS

from .regularization import process_geometry


def get_chunk_size(item_count: int, num_cores: int, max_size: int = 1000) -> int:
    """
    Calculate the chunk size for splitting a GeoDataFrame based on the number of items and cores.

    Parameters:
    -----------
    item_count : int
        The total number of items in the GeoDataFrame.
    num_cores : int
        The number of CPU cores available for processing.
    max_size : int, optional
        The maximum size of each chunk. Defaults to 1000.

    Returns:
    --------
    int
        The calculated chunk size.
    """
    # divide item_count by num_cores
    chunks_size = item_count // num_cores
    # make sure chunks_size is at least 1
    if chunks_size < 1:
        chunks_size = 1
    # make sure chunks_size is at most max_size
    if chunks_size > max_size:
        chunks_size = max_size
    return chunks_size


def split_gdf(gdf: gpd.GeoDataFrame, chunk_size: int) -> list[gpd.GeoDataFrame]:
    """
    Splits a GeoDataFrame into chunks of a specified size.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame to split.
    chunk_size : int
        The number of rows per chunk.

    Returns:
    --------
    list of geopandas.GeoDataFrame
        A list of GeoDataFrames, each containing up to `chunk_size` rows.
    """
    return [gdf[i : i + chunk_size] for i in range(0, len(gdf), chunk_size)]


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
    result_geodataframe = result_geodataframe[result_geodataframe.geometry.notna()]

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
    result_geodataframe: gpd.GeoDataFrame,
    target_crs: Optional[Union[str, pyproj.CRS]],
    simplify: bool,
    simplify_tolerance: float,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    allow_circles: bool,
    circle_threshold: float,
    include_metadata: bool,
):

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

    #  Add segments to avoid larger errors with big buildings
    result_geodataframe[result_geodataframe.geometry.name] = (
        result_geodataframe.geometry.segmentize(
            max_segment_length=simplify_tolerance * 5
        )
    )

    processed_data = result_geodataframe.geometry.apply(
        lambda geometry: process_geometry(
            geometry=geometry,
            parallel_threshold=parallel_threshold,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
            allow_circles=allow_circles,
            circle_threshold=circle_threshold,
            include_metadata=include_metadata,
        )
    )
    result_geodataframe["geometry"] = processed_data.apply(lambda x: x[0])
    if include_metadata:
        # Split the results into geometry and metadata columns
        result_geodataframe["iou"] = processed_data.apply(lambda x: x[1])
        result_geodataframe["main_direction"] = processed_data.apply(lambda x: x[2])

    # Clean up the resulting geometries (remove slivers)
    result_geodataframe = cleanup_geometry(
        result_geodataframe=result_geodataframe, simplify_tolerance=simplify_tolerance
    )

    # Reproject back to the original CRS if it was changed
    if target_crs is not None and original_crs is not None:
        # Check if CRS are actually different before reprojecting
        if not CRS.from_user_input(result_geodataframe.crs).equals(
            CRS.from_user_input(original_crs)
        ):
            result_geodataframe = result_geodataframe.to_crs(original_crs)
    return result_geodataframe
