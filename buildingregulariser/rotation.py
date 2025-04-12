import math
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.affinity import rotate


def rotate_point(
    point: np.ndarray,
    center: np.ndarray,
    angle_degrees: float,
) -> Tuple[float, float]:
    """
    Rotate a point clockwise around a center point

    Parameters:
    -----------
    point : array-like
        Point to rotate
    center : array-like
        Center of rotation
    angle_degrees : float
        Rotation angle in degrees

    Returns:
    --------
    tuple
        Rotated point coordinates
    """
    x, y = point
    center_x, center_y = center
    angle_radians = math.radians(angle_degrees)

    # Translate point to origin
    translated_x = x - center_x
    translated_y = y - center_y

    # Rotate
    rotated_x = translated_x * math.cos(angle_radians) + translated_y * math.sin(
        angle_radians
    )
    rotated_y = translated_y * math.cos(angle_radians) - translated_x * math.sin(
        angle_radians
    )

    # Translate back
    final_x = rotated_x + center_x
    final_y = rotated_y + center_y

    return (final_x, final_y)


def maximize_iou_by_rotation(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    max_rotation_degrees: int = 10,
    step_size: int = 1,
) -> gpd.GeoDataFrame:
    """
    Rotate geometries in gdf2 around their centroids to maximize
    Intersection over Union (IoU) with corresponding geometries in gdf1.

    Parameters:
    -----------
    gdf1 : gpd.GeoDataFrame
        First GeoDataFrame with geometries to compare against
    gdf2 : gpd.GeoDataFrame
        Second GeoDataFrame with geometries to be rotated
    max_rotation_degrees : int, default 5
        Maximum rotation in degrees (positive and negative)
    step_size : int, default 1
        Step size for the rotation angle search

    Returns:
    --------
    Tuple[gpd.GeoDataFrame, pd.Series]
        - A new GeoDataFrame with the rotated geometries
        - A Series containing the optimal rotation angles for each geometry
    """
    # Ensure the GeoDataFrames have the same length
    if len(gdf1) != len(gdf2):
        raise ValueError("Both GeoDataFrames must have the same number of rows")

    # Create a copy of gdf2 to store the results
    result_gdf = gdf2.copy()
    optimal_angles = pd.Series(index=gdf2.index, dtype=float)

    # Define the rotation angles to try
    rotation_angles = range(-max_rotation_degrees, max_rotation_degrees + 1, step_size)

    # Create a list to collect the best geometries
    best_geometries = []
    snapped_geoms = gdf2.rotation_snapped

    # Iterate through each pair of geometries
    for idx, (geom1, geom2, snapped) in enumerate(
        zip(gdf1.geometry, gdf2.geometry, snapped_geoms)
    ):
        if snapped:
            best_geometries.append(geom2)
            optimal_angles[idx] = 0
            continue
        max_iou = 0
        best_angle = 0
        best_rotated_geom = geom2

        # Get the centroid of geom2
        centroid = geom2.centroid

        # Try each rotation angle
        for angle in rotation_angles:
            # Rotate geom2 around its centroid
            rotated_geom = rotate(geom2, angle=angle, origin=centroid)

            # Calculate IoU
            intersection_area = geom1.intersection(rotated_geom).area
            union_area = geom1.union(rotated_geom).area

            # Handle edge case where union area is zero
            iou = 0 if union_area == 0 else intersection_area / union_area
            # print(f"Angle: {angle}, IoU: {iou}")

            # Update if we found a better rotation
            if iou > max_iou:
                max_iou = iou
                best_angle = angle
                best_rotated_geom = rotated_geom
        # Store the results
        best_geometries.append(best_rotated_geom)

        optimal_angles[idx] = best_angle

    result_gdf = result_gdf.set_geometry(
        gpd.GeoSeries(best_geometries, index=result_gdf.index, crs=result_gdf.crs)
    )

    return result_gdf
