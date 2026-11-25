from typing import List, Sequence

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


def polygons(geometry: BaseGeometry) -> List[Polygon]:
    """The non-empty polygons of a geometry."""
    return [
        p
        for p in shapely.get_parts(shapely.get_parts(geometry))
        if isinstance(p, Polygon) and not p.is_empty
    ]


def largest(geometry: BaseGeometry, min_area: float = 0.0) -> Polygon:
    """The largest polygon of a geometry if it has ``min_area``, else empty."""
    p = max(polygons(geometry), key=lambda p: p.area, default=Polygon())
    return p if p.area >= min_area else Polygon()


def polygonal(geometry: BaseGeometry) -> MultiPolygon:
    """The polygons of a geometry, valid (GEOS's buffer can return a bow-tie) and
    without the lines and points of a collapse."""
    return MultiPolygon(polygons(shapely.make_valid(geometry)))


def share_walls(geometries: Sequence[BaseGeometry], width: float) -> List[Polygon]:
    """
    Removes overlaps and closes narrow gaps between neighbouring polygons.

    Larger polygons are placed first. Each smaller one loses its overlap with the
    neighbours already placed and fills the gaps to them narrower than
    2 * ``width``, so touching buildings share one wall. A polygon left with less
    than (2 * ``width``) squared becomes empty.
    """
    grid = width / 1000  # snap-rounded overlays: robust where walls coincide
    out = [largest(shapely.set_precision(polygonal(g), grid)) for g in geometries]
    tree = shapely.STRtree(out)
    placed = np.zeros(len(out), dtype=bool)
    for i in np.argsort([-g.area for g in out], kind="stable"):
        near = tree.query(out[i], predicate="dwithin", distance=4 * width)
        placed[i] = True
        ref = [out[j] for j in near if placed[j] and j != i]
        if not ref:
            continue
        ref = polygonal(shapely.union_all(ref, grid_size=grid))
        core = largest(shapely.difference(out[i], ref, grid_size=grid), 4 * width**2)
        if core.is_empty:
            out[i] = core
            continue
        # A closing fills the gaps, kept within 2 * width of the polygon (it also
        # fills long slivers between nearly collinear walls).
        closed = shapely.union(out[i], ref, grid_size=grid)
        closed = polygonal(
            closed.buffer(width, join_style="mitre").buffer(-width, join_style="mitre")
        )
        reach = out[i].buffer(2 * width, join_style="mitre")
        gaps = shapely.intersection(closed, reach, grid_size=grid)
        grown = shapely.difference(polygonal(gaps), ref, grid_size=grid)
        inside = core.representative_point()
        out[i] = next((p for p in polygons(grown) if p.intersects(inside)), core)
    return out
