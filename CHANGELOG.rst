0.2.1:
=====
Fix bug with reprojection to utm:
- aeronet_raster - fix of utm south zone code 325XX -> 327XX
- aeronet_vector - add possibility to reproject to UTM from non-latlon crs

0.2.0:
=====
Change to aeronet_raster part (v.0.2.0).
CollectioProcessor now skips the source patches that contain only src_nodata values,
instead of processing_fn, the dst_nodata value is written to the output
Other changes to class signature:
- the CollectionProcessor does not accept kwargs anymore
- the args to pass to Writer are fixed (dst_nodata, src_nodata, dst_dtype)
- old params "nodata" and "dtype" can be specified, but they are deprecated

0.1.0:
======
Major refactoring:
 - divide functionality to 3 separate packages: independent `aeronet-raster` and `aeronet-vector` and depending on them `aeronet-convert`
 - preserve top-level aeronet package for legacy imports, so all previous public funcitons work the same
 - remove notebooks, criterions etc. that is not used
 - remove unused requirements
