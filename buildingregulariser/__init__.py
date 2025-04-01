"""
Polygon Regularization Package

A package for regularizing polygons by aligning edges to principal directions.
"""

from .__version__ import __version__
from .gdf_operations import regularize_geodataframe

# Package-wide exports
__all__ = [
    "regularize_geodataframe",
    "__version__",
]
