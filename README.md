# Building Footprint Regularization

A Python library for regularizing building footprints in geospatial data. This library helps clean up and standardize building polygon geometries by aligning edges to principal directions, correcting geometric irregularities.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Example Results

Before and after regularization:

![Before and After Regularization](/examples/1.png?raw=true "Before and After Regularization")


## Overview

Building footprints extracted from remote sensing imagery often contain noise, irregular edges, and geometric inconsistencies. This library provides tools to regularize these footprints by:

- Aligning edges to principal directions
- Converting near-rectangular buildings to perfect rectangles
- Simplifying complex polygons while maintaining their essential shape

Inspired by [RS-building-regularization](https://github.com/niecongchong/RS-building-regularization), this library takes a geometric approach to building regularization with improvements for usability and integration with the GeoPandas ecosystem.

## Installation

```bash
pip install buildingregulariser
```

## Quick Start

```python
import geopandas as gpd
from buildingregulariser import regularize_geodataframe

# Load your building footprints
buildings = gpd.read_file("buildings.gpkg")

# Regularize the building footprints
regularized_buildings = regularize_geodataframe(
    buildings, 
)

# Save the results
regularized_buildings.to_file("regularized_buildings.gpkg")
```

## Features

- **GeoDataFrame Integration**: Works seamlessly with GeoPandas GeoDataFrames
- **CRS Handling**: Intelligently handles coordinate reference systems
- **Polygon Regularization**: Aligns edges to principal directions
- **Edge Simplification**: Reduces the number of vertices while preserving shape
- **Geometry Cleanup**: Fixes invalid geometries and removes artifacts

## Usage Examples

### Basic Regularization

```python
from buildingregulariser import regularize_geodataframe
import geopandas as gpd

buildings = gpd.read_file("buildings.gpkg")
regularized = regularize_geodataframe(buildings)
```

### Working with Different Coordinate Systems

```python
regularized = regularize_geodataframe(
    buildings,
    target_crs="EPSG:3857"  # Web Mercator projection
)
```

### Fine-tuning Regularization Parameters

```python
regularized = regularize_geodataframe(
    buildings,
    parallel_threshold=2.0,   # Higher values allow more edge alignment
    simplify=True,
    simplify_tolerance=0.5    # Controls simplification level
)
```

#### Parameters:

- **geodataframe**: Input GeoDataFrame with polygon geometries
- **parallel_threshold**: Distance threshold for handling parallel lines (default: 1.0)
- **target_crs**: Target CRS for reprojection. If None, uses the input CRS
- **simplify**: If True, applies simplification to the geometry (default: True)
- **simplify_tolerance**: Tolerance for simplification (default: 0.5)

#### Returns:

- A new GeoDataFrame with regularized polygon geometries

## How It Works

1. **Edge Analysis**: Analyzes each polygon to identify principal directions
2. **Edge Orientation**: Aligns edges to be either parallel or perpendicular to the main direction
3. **Edge Connection**: Ensures proper connectivity between oriented edges


## License

This project is licensed under the MIT License

## Acknowledgments

This library was inspired by the [RS-building-regularization](https://github.com/niecongchong/RS-building-regularization) project, with improvements for integration with the GeoPandas ecosystem and enhanced regularization algorithms.