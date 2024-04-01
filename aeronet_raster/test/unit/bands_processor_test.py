from aeronet_raster.aeronet_raster.windowedprocessor import process_image
from aeronet_raster.aeronet_raster.dataadapters.rasterioadapter import RasterioWriter, RasterioReader
from aeronet_raster.aeronet_raster.dataadapters.separatebandsadapter import SeparateBandsWriter, SeparateBandsReader
import rasterio
import numpy as np
import logging
import cProfile
from pstats import SortKey, Stats
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

src = SeparateBandsReader([RasterioReader('test_data/RED.tif'),
                           RasterioReader('test_data/GRN.tif'),
                           RasterioReader('test_data/BLU.tif')])
profile = src.profile
dst = SeparateBandsWriter([RasterioWriter('test_data/RED_out.tif', **profile),
                           RasterioWriter('test_data/GRN_out.tif', **profile),
                           RasterioWriter('test_data/BLU_out.tif', **profile)])

#profile['count'] = 3
#dst = RasterioWriter('test_data/output.tif', **profile)
res = list()
import time
for _ in range(10):
    start = time.time()
    process_image(src, 1024, 256, processing, dst)
    res.append(time.time() - start)
print(res)
print(np.mean(res), np.std(res))
#cProfile.run('process_image(src, 1024, 256, processing, dst)', sort='tottime')



