from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import geopandas as gpd
from geopandas.sindex import SpatialIndex
from shapely.affinity import rotate


def process_row(
    idx_row: tuple,
    buffer_size: float,
    min_count: int,
    max_rotation: float,
    gdf_data: list,
    sindex_data: SpatialIndex,
) -> dict:
    """
    Processes a single polygon row for neighbor-based alignment.

    For each polygon, identifies neighboring polygons within a buffer distance,
    analyzes their directional distribution, and rotates the polygon to align
    with the dominant direction of its neighbors when appropriate criteria are met.

    Parameters:
    -----------
    idx_row : tuple
        A tuple containing (index, row) where index is the row index and
        row is a GeoDataFrame row with geometry and main_direction attributes.
    buffer_size : float
        Distance (in CRS units) used to identify neighboring polygons.
    min_count : int
        Minimum weighted count required for a direction to be considered dominant.
    max_rotation : float
        Maximum angular difference (in degrees) between a polygon's current direction
        and a neighbor direction to trigger alignment.
    gdf_data : list
        List of dictionaries containing serialized geometry and attribute data
        for all polygons in the dataset.
    sindex_data : SpatialIndex
        Spatial index of the GeoDataFrame for efficient spatial querying.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'idx': original row index
        - 'geometry': the original or aligned geometry
        - 'aligned_direction': the direction angle after alignment

    Notes:
    ------
    The function uses polygon perimeter as a weight when determining
    the dominant direction among neighbors, giving preference to
    larger neighboring polygons.
    """

    idx, row = idx_row
    geom = row.geometry
    search_geom = geom.buffer(buffer_size)

    # Use spatial index data for filtering
    candidate_idx = list(sindex_data.query(search_geom, predicate="intersects"))

    # Only do full geometric operations on the candidates
    neighbors_data = [
        gdf_data[i]
        for i in candidate_idx
        if search_geom.intersects(gdf_data[i]["geometry"])
    ]

    # Create a weighted dictionary of directions
    direction_weights = defaultdict(float)

    # Add weights for each direction and its perpendicular counterpart
    for neighbor in neighbors_data:
        direction = neighbor["main_direction"]
        weight = neighbor["perimeter"]

        # Add weight for the original direction
        direction_weights[direction] += weight

        # Add weight for the perpendicular direction
        perp_direction = 90 - direction
        direction_weights[perp_direction] += weight

    # Sort directions by their weights (highest first)
    sorted_directions = sorted(
        direction_weights.items(), key=lambda x: x[1], reverse=True
    )

    # Find the best direction to align with
    result = {
        "idx": idx,
        "geometry": row.geometry,
        "aligned_direction": row.main_direction,
    }

    for align_dir, weight in sorted_directions[:4]:
        direction_delta = row.main_direction - align_dir
        if weight > min_count and abs(direction_delta) <= max_rotation:
            result["aligned_direction"] = align_dir
            result["geometry"] = rotate(
                row.geometry, -direction_delta, origin="centroid"
            )
            break

    return result


def align_with_neighbor_polygons(
    gdf: gpd.GeoDataFrame,
    buffer_size: float = 350.0,
    min_count: int = 3,
    max_rotation: float = 10,
    include_metadata: bool = False,
    num_cores: int = -1,
) -> gpd.GeoDataFrame:
    """
    Align polygon orientations based on their neighbors using multiprocessing.imap.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygons and main_direction column
    buffer_size : float
        Distance to consider for neighboring polygons
    min_count : int
        Minimum weight required for direction consideration
    max_rotation : float
        Maximum angular difference to trigger alignment
    include_metadata : bool
        Whether to include metadata columns in output
    num_cores : int
        Number of parallel processes (-1 for all available)

    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with aligned polygons
    """
    # Create a copy and add necessary columns
    result_gdf = gdf.copy()
    result_gdf["aligned_direction"] = result_gdf["main_direction"].copy()
    result_gdf["perimeter"] = result_gdf.geometry.length

    # Ensure spatial index exists
    if not result_gdf.sindex:
        result_gdf = result_gdf.copy()

    # Prepare serializable data for workers
    gdf_data = [
        {
            "geometry": row.geometry,
            "main_direction": row.main_direction,
            "perimeter": row.perimeter,
        }
        for _, row in result_gdf.iterrows()
    ]

    # Get spatial index
    sindex = result_gdf.sindex

    # Prepare arguments for each row
    idx_rows = [(idx, row) for idx, row in result_gdf.iterrows()]

    # Process in parallel using imap
    results = []
    process_row_partial = partial(
        process_row,
        buffer_size=buffer_size,
        min_count=min_count,
        max_rotation=max_rotation,
        gdf_data=gdf_data,
        sindex_data=sindex,
    )
    with Pool(processes=num_cores) as pool:
        # Use imap for better memory efficiency with large datasets
        for result in pool.imap(process_row_partial, idx_rows, chunksize=100):
            results.append(result)

    # Update the GeoDataFrame with results
    for result in results:
        idx = result["idx"]
        result_gdf.at[idx, "geometry"] = result["geometry"]
        result_gdf.at[idx, "aligned_direction"] = result["aligned_direction"]

    # Clean up if needed
    if not include_metadata:
        result_gdf = result_gdf.drop(columns=["aligned_direction", "perimeter"])

    return result_gdf
