import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box

from buildingregulariser import regularize_geodataframe


def overlap(geoms: list) -> float:
    tree = shapely.STRtree(geoms)
    pairs = zip(*tree.query(geoms, predicate="intersects"), strict=True)
    return sum(geoms[i].intersection(geoms[j]).area for i, j in pairs if i < j)


def test_touching_buildings_share_walls_without_overlap():
    """Three 9 x 9 m houses in a row, polygonized from a 0.15 m label raster: each
    has a ragged roof, so they are regularized differently along the shared walls."""
    rng = np.random.default_rng(0)
    houses = []
    for x0 in (0.0, 9.0, 18.0):
        roof = [box(x, 9, x + 0.15, 9.15) for x in x0 + 0.15 * np.arange(60)]
        roof = [c for c in roof if rng.integers(0, 2)]
        houses.append(shapely.union_all([box(x0, 0, x0 + 9, 9), *roof]))
    gdf = gpd.GeoDataFrame({"name": ["a", "b", "c"]}, geometry=houses, crs=3857)

    out = regularize_geodataframe(
        gdf, num_cores=1, coverage=True, simplify_tolerance=0.3
    )
    geoms = list(out.geometry)
    assert list(out["name"]) == ["a", "b", "c"]
    assert out.geometry.is_valid.all()
    assert overlap(geoms) < 1e-6
    for a, b in zip(geoms, geoms[1:], strict=False):
        assert a.boundary.intersection(b.boundary).length > 8


def test_overlap_goes_to_the_larger_building():
    gdf = gpd.GeoDataFrame(geometry=[box(8, 2, 14, 8), box(0, 0, 10, 10)], crs=3857)
    out = regularize_geodataframe(gdf, num_cores=1, coverage=True)
    assert overlap(list(out.geometry)) < 1e-6
    assert abs(out.geometry.iloc[1].area - 100) < 1
    assert abs(out.geometry.iloc[0].area - 24) < 1
