from aeronet_raster.aeronet_raster.dataadapters.rasterioadapter import RasterioReader, RasterioWriter
import numpy as np
import rasterio

# Writes some random data into the file and reads it back
# both writing and reading occurs partially out of bounds (top-left window coord is -4, -4)

profile = {'width': 256,
           'height': 256,
           'dtype': 'uint8',
           'count': 3,
           'driver': 'GTiff',
           'transform': rasterio.Affine(10.0, 0, 100000, 0, -10.0, 20000),
           'crs': 'EPSG:3857',
           'nodata': 0}

random_array = np.random.randint(0, 255, size=(3, 8, 8))
print(random_array.shape)
RasterioWriter('src.tif', **profile)[:, -4:4, -4:4] = random_array
read_array = RasterioReader('src.tif')[:, -4:4, -4:4]
print(read_array.shape)
print(read_array == random_array)  # bottom-right corner must be true

