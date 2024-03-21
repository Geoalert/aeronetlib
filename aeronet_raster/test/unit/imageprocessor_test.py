from aeronet_raster.aeronet_raster.windowedprocessor import process_image
from aeronet_raster.aeronet_raster.dataadapters.rasterioadapter import RasterioWriter, RasterioReader
import rasterio
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

profile = {'width': 16, 'height': 16, 'count': 3, 'dtype': 'uint8' }

random_data = np.random.randint(0, 254, size=(3, 16, 16))
processing = lambda x: x+1
with rasterio.open('tst.tif', 'w', **profile) as dst:  # write random file
    dst.write(random_data)

# case1 - equal window size, no margin
print('Case1')
process_image(RasterioReader('tst.tif'), 4, 0, processing, RasterioWriter('res.tif', **profile), verbose=True)
with rasterio.open('res.tif') as d:
    read_data = d.read()
print(read_data == processing(random_data))

# case2 - equal window size, margin
print('Case2')
process_image(RasterioReader('tst.tif'), 8, 2, processing, RasterioWriter('res2.tif', **profile), verbose=True)
with rasterio.open('res2.tif') as d:
    read_data = d.read()
print(read_data == processing(random_data))


# case3 - unequal window size, margin. Reading from (3, 16, 16) with sample_size=4,
# then processing_fn reduce each sample from (3, 8, 8) to (1, 1, 1) and we write it into (1, 4, 4)
print('Case3')
dst_profile = {'width': 4, 'height': 4, 'count': 1, 'dtype': 'uint8'}
processing = lambda x: np.mean(x).reshape(1, 1, 1).astype(int)
process_image(RasterioReader('tst.tif'),
              src_sample_size=4,
              src_margin=0,
              processor=processing,
              dst=RasterioWriter('res3.tif', **dst_profile),
              dst_sample_size=1,
              dst_margin=0,
              verbose=True)
with rasterio.open('res3.tif') as d:
    read_data = d.read()

# calculating true result to compare
gt = np.zeros((1, 4, 4))
for i in range(4):
    for j in range(4):
        gt[0, i, j] = int(np.mean(random_data[:, i*4:(i+1)*4, j*4:(j+1)*4]))
print(random_data)
print(read_data)
print(gt)
print(read_data == gt)
