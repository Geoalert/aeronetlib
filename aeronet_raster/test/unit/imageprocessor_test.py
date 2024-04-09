from aeronet_raster.aeronet_raster import dataprocessor
from aeronet_raster.aeronet_raster.dataadapters import rasterioadapter
import rasterio
import numpy as np
import logging

#logging.basicConfig(level=logging.INFO)

with rasterioadapter.RasterioReader('test_data/input.tif') as src:
    profile = src.profile
    with rasterioadapter.RasterioWriter('test_data/output.tif', profile) as dst:
        dataprocessor.process_image(src=src,
                                    src_sample_size=512,
                                    src_margin=64,
                                    processor=lambda x: x,
                                    dst=dst,
                                    mode='crop',
                                    verbose=True)