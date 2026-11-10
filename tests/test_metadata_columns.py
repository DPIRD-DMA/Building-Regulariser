"""Which columns regularize_geodataframe returns, with and without metadata."""

import geopandas as gpd
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

from buildingregulariser import regularize_geodataframe

METADATA_COLS = {"iou", "main_direction", "perimeter", "aligned_direction"}


def _buildings() -> gpd.GeoDataFrame:
    """Three slightly rotated rectangles close enough to be neighbours."""
    base = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
    geoms = [
        translate(rotate(base, angle, origin="centroid"), xoff=offset)
        for angle, offset in [(3, 0), (-4, 20), (6, 40)]
    ]
    return gpd.GeoDataFrame(
        {"building_id": [101, 102, 103], "name": ["a", "b", "c"]},
        geometry=geoms,
        crs="EPSG:32750",
    )


def _run(include_metadata: bool, neighbor_alignment: bool) -> gpd.GeoDataFrame:
    return regularize_geodataframe(
        _buildings(),
        num_cores=1,
        include_metadata=include_metadata,
        neighbor_alignment=neighbor_alignment,
        neighbor_search_distance=50,
    )


@pytest.mark.parametrize("neighbor_alignment", [False, True])
def test_metadata_columns_dropped_when_not_requested(neighbor_alignment):
    out = _run(include_metadata=False, neighbor_alignment=neighbor_alignment)
    assert METADATA_COLS.isdisjoint(out.columns)


@pytest.mark.parametrize(
    "neighbor_alignment,expected",
    [
        (False, {"iou", "main_direction"}),
        (True, METADATA_COLS),
    ],
)
def test_metadata_columns_present_when_requested(neighbor_alignment, expected):
    out = _run(include_metadata=True, neighbor_alignment=neighbor_alignment)
    assert expected <= set(out.columns)
    assert out[list(expected)].notna().all().all()


@pytest.mark.parametrize("include_metadata", [False, True])
@pytest.mark.parametrize("neighbor_alignment", [False, True])
def test_user_columns_preserved(include_metadata, neighbor_alignment):
    out = _run(include_metadata=include_metadata, neighbor_alignment=neighbor_alignment)
    assert list(out["building_id"]) == [101, 102, 103]
    assert list(out["name"]) == ["a", "b", "c"]
    assert out.geometry.name == "geometry"
