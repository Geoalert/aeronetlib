from aeronet_raster.aeronet_raster.windowedprocessor import process_image
from aeronet_raster.aeronet_raster.dataadapters.rasterioadapter import RasterioWriter, RasterioReader
from aeronet_raster.aeronet_raster.dataadapters.separatebandsadapter import SeparateBandsWriter, SeparateBandsReader
import rasterio
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

profile = {'width': 16, 'height': 16, 'count': 3, 'dtype': 'uint8' }

random_data = np.random.randint(0, 254, size=(3, 16, 16))
processing = lambda x: x+1

profile['count'] = 1
with rasterio.open('RED.tif', 'w', **profile) as dst:
    dst.write(random_data[0], 1)
with rasterio.open('GRN.tif', 'w', **profile) as dst:
    dst.write(random_data[1], 1)
with rasterio.open('BLU.tif', 'w', **profile) as dst:
    dst.write(random_data[2], 1)

src = SeparateBandsReader([RasterioReader('RED.tif'), RasterioReader('GRN.tif'), RasterioReader('BLU.tif')])
dst = SeparateBandsWriter([RasterioWriter('RED_res.tif', **profile),
                           RasterioWriter('GRN_res.tif', **profile),
                           RasterioWriter('BLU_res.tif', **profile)])

process_image(src, 8, 2, processing, dst, verbose=True)

del src
del dst

res = list()
with rasterio.open('RED_res.tif') as d:
    res.append(d.read(1))
with rasterio.open('GRN_res.tif') as d:
    res.append(d.read(1))
with rasterio.open('BLU_res.tif') as d:
    res.append(d.read(1))

print(res == processing(random_data))
