import geopandas as gpd

from .regularization import process_geometry


def get_chunk_size(item_count: int, num_cores: int, max_size: int = 1000) -> int:
    """
    Calculate an optimal chunk size for parallel processing.

    Determines a balanced chunk size based on the number of items to process
    and available CPU cores, while respecting a maximum chunk size limit.

    Parameters:
    -----------
    item_count : int
        Total number of items to be processed.
    num_cores : int
        Number of CPU cores available for parallel processing.
    max_size : int, optional
        Maximum allowed chunk size regardless of other factors. Default is 1000.

    Returns:
    --------
    int
        Calculated chunk size, at least 1, at most max_size, and balanced
        across available cores.
    """
    # Calculate initial chunk size
    chunk_size = (item_count + num_cores - 1) // num_cores

    # Ensure it's not too large
    if chunk_size > max_size:
        chunk_size = max_size

    # Ensure it's at least 1
    return max(1, chunk_size)


def split_gdf(gdf: gpd.GeoDataFrame, chunk_size: int) -> list[gpd.GeoDataFrame]:
    """
    Splits a GeoDataFrame into smaller chunks for parallel processing.

    Divides a GeoDataFrame into a list of smaller GeoDataFrames, each
    containing at most chunk_size rows. This facilitates parallel processing
    of large datasets by distributing work across multiple processes.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to split into chunks.
    chunk_size : int
        Maximum number of rows in each chunk.

    Returns:
    --------
    list[gpd.GeoDataFrame]
        A list of GeoDataFrame chunks, each containing at most chunk_size rows.
        The original index values are preserved in each chunk.

    Notes:
    ------
    Empty GeoDataFrames will result in an empty list.
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


def process_geometry_wrapper(
    result_geodataframe: gpd.GeoDataFrame,
    simplify: bool,
    simplify_tolerance: float,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    allow_circles: bool,
    circle_threshold: float,
    include_metadata: bool,
):
    """
    This wrapper function coordinates the full process of polygon regularization by:
    1. Applying initial simplification to reduce vertex count (if requested)
    2. Segmenting complex geometries to maintain fidelity during processing
    3. Applying the regularization algorithm to each geometry
    4. Extracting and storing metadata about the regularization process (if requested)
    5. Performing geometry cleanup to remove artifacts and slivers

    Parameters:
    -----------
    result_geodataframe : gpd.GeoDataFrame
        The GeoDataFrame containing geometries to be regularized.
    simplify : bool
        Whether to apply initial simplification to reduce vertex count before regularization.
    simplify_tolerance : float
        Tolerance value for simplification operations, in the same units as the GeoDataFrame's CRS.
        Also affects segment size during geometry preparation.
    parallel_threshold : float
        Distance threshold for merging nearly parallel adjacent edges during regularization.
    allow_45_degree : bool
        If True, allows edges to be oriented at 45-degree angles relative to the main direction.
    diagonal_threshold_reduction : float
        Reduction factor (in degrees) to decrease the likelihood of diagonal edges being created.
        Values range from 0 to 22.5 degrees; higher values reduce diagonal edges.
    allow_circles : bool
        If True, attempts to detect nearly circular polygons and replaces them with perfect circles.
    circle_threshold : float
        Intersection over Union (IoU) threshold for circle detection. Values range from 0 to 1,
        with higher values requiring closer resemblance to a perfect circle.
    include_metadata : bool
        If True, includes metadata columns in the output GeoDataFrame:
        - 'iou': Intersection over Union between original and regularized geometries
        - 'main_direction': Principal direction angle (degrees) used for regularization

    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame with regularized geometries and optional metadata columns.
        Invalid or empty geometries resulting from processing are removed.
    """
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

    return result_geodataframe
