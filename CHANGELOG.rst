0.1.0:
======
Major refactoring:
 - divide functionality to 3 separate packages: independent `aeronet-raster` and `aeronet-vector` and depending on them `aeronet-convert`
 - preserve top-level aeronet package for legacy imports, so all previous public funcitons work the same
 - remove notebooks, criterions etc. that is not used
 - remove unused requirements
