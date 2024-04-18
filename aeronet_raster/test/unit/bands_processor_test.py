from aeronet_raster.aeronet_raster.dataprocessor import process_image
from aeronet_raster.aeronet_raster.dataadapters.separatebandsadapter import from_rasterio_bands
import rasterio
import numpy as np
import logging
import time
logging.basicConfig(level=logging.INFO)


def create_random_bands():
    profile = {'width': 16, 'height': 16, 'count': 3, 'dtype': 'uint8' }
    random_data = np.random.randint(0, 254, size=(3, 16, 16))
    profile['count'] = 1
    with rasterio.open('test_data/RED.tif', 'w', **profile) as dst:
        dst.write(random_data[0], 1)
    with rasterio.open('test_data/GRN.tif', 'w', **profile) as dst:
        dst.write(random_data[1], 1)
    with rasterio.open('test_data/BLU.tif', 'w', **profile) as dst:
        dst.write(random_data[2], 1)
    return random_data, profile


def validate_result(random_data):
    res = list()
    with rasterio.open('RED_res.tif') as d:
        res.append(d.read(1))
    with rasterio.open('GRN_res.tif') as d:
        res.append(d.read(1))
    with rasterio.open('BLU_res.tif') as d:
        res.append(d.read(1))
    print(res == processing(random_data))


processing = lambda x: x


src = from_rasterio_bands(('test_data/RED.tif', 'test_data/GRN.tif', 'test_data/BLU.tif'))
profile = src.profile
dst = from_rasterio_bands(('test_data/RED_out.tif', 'test_data/GRN_out.tif', 'test_data/BLU_out.tif'), 'w', profile)

res = list()
for _ in range(10):
    start = time.time()
    process_image(src, 1024, 256, processing, dst)
    res.append(time.time() - start)
print(res)
print(np.mean(res), np.std(res))
