import numpy as np
from aeronet_raster.aeronet_raster import dataprocessor
from aeronet_raster.aeronet_raster.dataadapters import numpyadapter

inp = numpyadapter.NumpyReader(np.zeros((1, 30, 30)))
out = numpyadapter.NumpyWriter(np.zeros((1, 30, 30)))

dataprocessor.process_image(src = inp,
                            src_sample_size = 10,
                            src_margin = 4,
                            processor = lambda x: x+np.random.rand(),
                            dst = out,
                            dst_sample_size = 10,
                            dst_margin = 4,
                            dst_margin_mode = 'crossfade')
#plt.imshow(out._data[0])
