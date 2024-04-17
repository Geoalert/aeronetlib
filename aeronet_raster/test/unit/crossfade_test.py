import numpy as np
from aeronet_raster.aeronet_raster import dataprocessor
from aeronet_raster.aeronet_raster.dataadapters import rasterioadapter
import logging
logging.basicConfig(level=logging.INFO)

with rasterioadapter.RasterioReader('test_data/input2.tif') as src:
    profile = src.profile
    profile['dtype'] = 'float32'
    with rasterioadapter.RasterioWriter('test_data/output.tif', profile) as dst:
        dataprocessor.process_image(src=src,
                                    src_sample_size=512,
                                    src_margin=64,
                                    processor=lambda x: x,#np.ones((3, 512, 512))*127,
                                    dst=dst,
                                    mode='crossfade',
                                    verbose=True)

