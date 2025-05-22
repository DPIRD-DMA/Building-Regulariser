from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.affinity import rotate


def process_row(
    idx: int,
    buffer_size: float,
    max_rotation: float,
    gdf: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """
    Aligns a single polygon's orientation with its neighbours if a dominant direction is detected.

    For a given polygon index in the GeoDataFrame, this function identifies neighbouring polygons
    within a specified buffer, aggregates their directional data weighted by perimeter, and, if
    conditions are met, rotates the polygon to align with the dominant neighbour direction.

    Parameters:
    -----------
    idx : int
        Index of the polygon row in the GeoDataFrame.
    buffer_size : float
        Distance used to define the neighbourhood search area around the polygon.
    max_rotation : float
        Maximum allowed rotation (in degrees) from the current to the proposed direction.
    gdf : gpd.GeoDataFrame
        The full GeoDataFrame containing all polygons and required attributes:
        - 'geometry': polygon geometry
        - 'main_direction': original orientation angle
        - 'perimeter': polygon perimeter (used as weight)

    Returns:
    --------
    dict
        A dictionary with:
        - 'idx': the index of the processed row
        - 'geometry': original or rotated polygon geometry
        - 'aligned_direction': selected direction used for alignment
    """
    row = gdf.iloc[idx]
    geom = row.geometry
    search_geom = geom.buffer(buffer_size)

    # Use spatial index data for filtering
    candidate_idx = gdf.sindex.query(search_geom, predicate="intersects")

    # Only do full geometric operations on the candidates
    neighbors_data = gdf.iloc[candidate_idx]

    # Combine original and perpendicular directions into one Series
    all_directions = pd.concat(
        [neighbors_data["main_direction"], 90 - neighbors_data["main_direction"]]
    )
    # Calculate weights based on perimeter
    all_weights = pd.concat([neighbors_data["perimeter"], neighbors_data["perimeter"]])

    # Aggregate weights
    grouped_weights = all_weights.groupby(all_directions).sum()

    # Convert to defaultdict
    direction_weights = defaultdict(float, grouped_weights.to_dict())

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

    for align_dir, _ in sorted_directions[:4]:
        direction_delta = row.main_direction - align_dir
        if abs(direction_delta) <= max_rotation:
            result["aligned_direction"] = align_dir
            result["geometry"] = rotate(
                row.geometry, -direction_delta, origin="centroid"
            )
            break

    return result


def align_with_neighbor_polygons(
    gdf: gpd.GeoDataFrame,
    num_cores: int,
    buffer_size: float,
    max_rotation: float,
    include_metadata: bool,
) -> gpd.GeoDataFrame:
    """
    Aligns the orientation of polygons in a GeoDataFrame based on their neighbors' dominant direction.

    Each polygon is evaluated in parallel. A buffer is used to identify neighboring polygons,
    which are then used to infer a dominant direction. If a suitable direction is found within
    a defined angular threshold, the polygon is rotated to match it.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with 'geometry' and 'main_direction' columns.
    num_cores : int
        Number of processes to use for parallel processing
    buffer_size : float, default=350.0
        Buffer distance for determining neighborhoods.
    max_rotation : float, default=10
        Maximum rotation angle allowed for alignment (in degrees).
    include_metadata : bool, default=False
        Whether to retain intermediate columns such as 'aligned_direction' and 'perimeter'.

    Returns:
    --------
    gpd.GeoDataFrame
        A copy of the original GeoDataFrame with aligned geometries. Intermediate metadata columns
        are included only if `include_metadata` is True.
    """
    # Create a copy and add necessary columns
    gdf = gdf.explode(ignore_index=True).copy()
    gdf["aligned_direction"] = gdf["main_direction"].copy()
    gdf["perimeter"] = gdf.geometry.length

    # Process in parallel using imap
    results = []
    process_row_partial = partial(
        process_row,
        buffer_size=buffer_size,
        max_rotation=max_rotation,
        gdf=gdf,
    )

    # Work out chunksize for imap
    row_count = len(gdf)
    chunksize = min(max(row_count // num_cores, 1), 5000)

    with Pool(processes=num_cores) as pool:
        results = pool.map(process_row_partial, range(len(gdf)), chunksize=chunksize)

    # Update the GeoDataFrame with results
    for result in results:
        idx = result["idx"]
        gdf.at[idx, "geometry"] = result["geometry"]
        gdf.at[idx, "aligned_direction"] = result["aligned_direction"]

    # Clean up if needed
    if not include_metadata:
        gdf = gdf.drop(columns=["aligned_direction", "perimeter"])

    return gdf
