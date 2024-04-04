from aeronet_raster.aeronet_raster.dataadapters import piladapter, separatebandsadapter
import numpy as np

with separatebandsadapter.SeparateBandsReader([piladapter.PilReader(b) for b in ('test_data/image.png',
                                                                                 'test_data/image2.png')]) as d:
    print(d.shape)  # since image.png and image2.png both have 3 channels, resulting file will have 6 channels

    print(d[2:5, -50:200, -50:200].shape)