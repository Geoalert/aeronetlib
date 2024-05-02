from aeronet_raster.aeronet_raster import CollectionProcessor, BandCollection
import cProfile
import numpy as np
import time
start = time.time()

processor = CollectionProcessor(input_channels=['RED', 'GRN', 'BLU'],
                                output_labels=['RED_out', 'GRN_out', 'BLU_out'],
                                processing_fn=lambda x: x,
                                n_workers=0,
                                sample_size=(512, 512),
                                bound=256)

bc = BandCollection(['test_data/RED.tif', 'test_data/GRN.tif', 'test_data/BLU.tif'])
#cProfile.run('labels_bc = processor.process(bc, "test_data")', sort='tottime')

res = list()
for _ in range(10):
    start = time.time()
    processor.process(bc, "test_data")
    res.append(time.time() - start)
print(res)
print(np.mean(res), np.std(res))

