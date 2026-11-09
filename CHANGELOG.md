# Changelog

All notable changes to Building-Regulariser are documented here.

## [Unreleased]

### Fixed
- Removed the `inplace=True` call to `GeoDataFrame.drop`, which pandas 3
  deprecates (`Pandas4Warning`) and pandas 4 will remove.

## [0.2.5] - 2026-05-04

### Added
- Pre-commit configuration (`ruff-check`, `ruff-format`, and a pre-push
  `pytest` hook).
- GitHub Actions CI workflow running ruff lint, mypy, and pytest on
  every push and pull request.
- GitHub Actions publish workflow that builds the package and pushes to
  PyPI via OIDC trusted publishing on `v*` tag pushes.
- `py.typed` marker so downstream type checkers pick up the package's
  type hints.
- Multipolygon regression tests covering self-intersecting and
  vertex-touching inputs.

### Changed
- Versioning is now derived from git tags via `setuptools-scm`. The
  hardcoded `__version__.py` has been removed; `__version__` is now
  read at runtime from package metadata via `importlib.metadata`.
- `regularize_single_polygon` signature widened to
  `Polygon | MultiPolygon` to match its actual runtime behaviour.
- `uv.lock` is no longer tracked in version control.

### Fixed
- Type errors in `geometry_utils.rotate_edge` where the no-rotation
  branch returned `ndarray` instead of the expected coordinate tuple.

## [0.2.4] - 2025-07-24

### Added
- Project metadata: license, keywords, project URL.
- Ruff to the dev dependency group.

### Fixed
- Improved robustness when input geometries are invalid
  (self-intersecting or otherwise malformed polygons).
- Handling of invalid inputs reported in issue #5.

### Changed
- Layer order in the regularization pipeline.
- Updated example parameters and dataset.

## [0.2.2] - 2025-05-22

### Added
- Neighbour alignment: edges of nearby buildings can be aligned to a
  shared direction.
- End-to-end tests for geometry quality and parameterized regularization.
- Example notebook and data.

### Changed
- Throughput optimisations.
- Improved spatial-index handling and data preparation in neighbour
  alignment.
- Internal consolidation and refactoring; `rotate_edge` and other line
  operations consolidated into `geometry_utils`.
- Type-hint and argument-name cleanup across the codebase.

## [0.1.12] - 2025-04-12

### Added
- `include_metadata` option to return per-feature regularization
  metadata alongside the output geometries.
- Conda-forge install instructions.

### Changed
- Improved main-edge finding with mirroring and smoothing.
- Relaxed dependency version requirements.

## [0.1.11] - 2025-04-10

### Changed
- Reordered filter operations in the regularization pipeline.

## [0.1.10] - 2025-04-09

### Added
- Coarse and fine bins when finding the main direction of a polygon.

## [0.1.9] - 2025-04-09

### Added
- Per-feature metadata output.
- Inputs are now exploded so multipart geometries are handled.
- Increased histogram bin size for direction estimation.

## [0.1.8] - 2025-04-04

### Added
- Additional histogram bins for finer direction estimation.

## [0.1.7] - 2025-04-03

### Added
- Exposed the diagonal threshold reduction parameter.

## [0.1.6] - 2025-04-03

### Fixed
- Handling of unusually large buildings.

## [0.1.4] - 2025-04-02

### Added
- Optional circle output for round-shaped buildings.

### Changed
- Throughput improvements.

## [0.1.3] - 2025-04-01

### Added
- MIT license file and project README.

### Fixed
- Cleaning logic refinements following the initial release.

## [0.1.0] - 2025-04-01

### Added
- Initial public release with the core `regularize_geodataframe` API
  for aligning building polygon edges to principal directions.

[Unreleased]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.2.5...HEAD
[0.2.5]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.2.2...v0.2.4
[0.2.2]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.12...v0.2.2
[0.1.12]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.4...v0.1.6
[0.1.4]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/DPIRD-DMA/Building-Regulariser/compare/v0.1.0...v0.1.3
[0.1.0]: https://github.com/DPIRD-DMA/Building-Regulariser/releases/tag/v0.1.0
