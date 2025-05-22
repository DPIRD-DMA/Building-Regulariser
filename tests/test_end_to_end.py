from pathlib import Path
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry.base import BaseGeometry

from buildingregulariser import regularize_geodataframe

cwd = Path(__file__).parent
OUTPUT_FILE = cwd.parent / "test data/output/test output.gpkg"
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()
INPUT_FILE = cwd.parent / "test data/input/test_data.gpkg"
DEFAULT_PARAMS = dict(
    parallel_threshold=1.0,
    simplify=True,
    simplify_tolerance=0.5,
    allow_45_degree=True,
    diagonal_threshold_reduction=15,
    allow_circles=True,
    circle_threshold=0.9,
    num_cores=1,
    include_metadata=False,
    neighbor_alignment=False,
    neighbor_search_distance=100,
    neighbor_max_rotation=10,
)

assert INPUT_FILE.exists(), f"Test file missing: {INPUT_FILE}"
input_gdf = gpd.read_file(INPUT_FILE)


# --- Geometry Quality Checks ---
def iou(poly1: BaseGeometry, poly2: BaseGeometry) -> float:
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union != 0 else 0


def check_geometry_quality(inputs, outputs, iou_threshold=0.4, perimeter_tolerance=0.6):
    assert not outputs.empty
    assert outputs.geometry.is_valid.all()
    assert outputs.geometry.notnull().all()
    assert len(inputs) == len(
        outputs
    ), f"Row count mismatch: {len(inputs)} != {len(outputs)}"

    for idx, (i_geom, o_geom) in enumerate(zip(inputs.geometry, outputs.geometry)):
        assert i_geom.intersects(o_geom), f"No intersection for feature {idx}"

        overlap_iou = iou(i_geom, o_geom)
        assert (
            overlap_iou >= iou_threshold
        ), f"Low IoU for feature {idx}: {overlap_iou:.2f}"

        in_perim = i_geom.length
        out_perim = o_geom.length
        min_perim = in_perim * (1 - perimeter_tolerance)
        max_perim = in_perim * (1 + perimeter_tolerance)

        assert (
            min_perim <= out_perim <= max_perim
        ), f"Perimeter out of range for feature {idx}: {out_perim:.2f} (expected {min_perim:.2f}–{max_perim:.2f})"


# --- Parametrized Tests ---
@pytest.mark.parametrize(
    "param,values",
    [
        ("parallel_threshold", [0.5, 2.0, 5.0]),
        ("simplify", [True, False]),
        ("simplify_tolerance", [0.3, 1.0, 3.0]),
        ("allow_45_degree", [True, False]),
        ("diagonal_threshold_reduction", [0, 22.5, 45]),
        ("allow_circles", [True, False]),
        ("circle_threshold", [0.5, 0.75, 0.99]),
        ("num_cores", [0, 1, 4]),
        ("include_metadata", [False, True]),
        ("neighbor_alignment", [False, True]),
        ("neighbor_search_distance", [0, 100, 350]),
        ("neighbor_max_rotation", [0, 22.5, 45]),
    ],
)
def test_regularize_param_variants(param, values):
    for val in values:
        config: dict[str, Any] = DEFAULT_PARAMS.copy()
        if param in {
            "simplify",
            "allow_45_degree",
            "allow_circles",
            "include_metadata",
            "neighbor_alignment",
        }:
            config[param] = bool(val)
        elif param in {"num_cores"}:
            config[param] = int(val)
        else:
            config[param] = val
        result = regularize_geodataframe(geodataframe=input_gdf.copy(), **config)
        layer_name = f"{param}_{str(val).replace('.', '_')}"
        result.to_file(OUTPUT_FILE, layer=layer_name, driver="GPKG")
        print(f"Saved layer '{layer_name}' to {OUTPUT_FILE}")
        check_geometry_quality(input_gdf, result)
