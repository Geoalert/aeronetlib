from aeronet_raster import BandCollection, parse_directory, CollectionProcessor


def processing_fn(sample):
    out = sample.numpy()
    return out


def read_bc(path, bands):
    bands = parse_directory(path, bands)
    return BandCollection(bands)


predictor = CollectionProcessor(
    input_channels=['RED', 'GRN', 'BLU'],
    output_labels=['RED1', 'GRN1', 'BLU1'],
    processing_fn=processing_fn,
    n_workers=1,
    sample_size=(1024, 1024),
    bound=256,
    verbose=True,
    dtype='uint8',
    bound_mode='weight'
)
path = '/home/adeshkin/projects/urban-local-pipeline/data/head_style_transfer_pda_vs_cyclegan/aze_samples/1/input'
bc = read_bc(path, ['RED', 'GRN', 'BLU'])
predictor.process(bc, path)

import rasterio
import numpy as np

with rasterio.open(f'{path}/BLU1.tif') as f:
    img1 = f.read().astype(np.uint8)

with rasterio.open(f'{path}/BLU.tif') as f:
    img = f.read()
    kwargs = f.meta.copy()
d = (np.abs(img - img1) > 0)[0]
y_inds, x_inds = np.nonzero(d)
print(len(y_inds), len(x_inds))
print(y_inds[:5], y_inds[-5:])
print(x_inds[:5], x_inds[-5:])
print(d.shape)
# print(n
# p.abs(np.unique(img - img1)))
# print(d.shape)
# print(d.sum())
# print(d.sum() /(img.shape[1] * img.shape[2]))
# print(np.unique(d))
# with rasterio.open(f'{path}/BLU_diff.tif', 'w', **kwargs) as f:
#     f.write(d, 1)
#
# import numpy as np
# # bound = 128
# #
# # wx1 = [x/(2*bound) for x in range(2*bound)]
# # wx2 = [x for x in range(window, window + 2*bound)]
#
# def _get_weight_item( x, window, bound):
#     if x < 2 * bound:
#         w_x = x / (2 * bound)
#     elif 2 * bound <= x < window:
#         w_x = 1
#     else:  # i >= window:
#         w_x = (window + 2 * bound - x) / (2 * bound)
#
#     return w_x
#
#
# def _get_weight_mtrx(sample_size, bound):
#     mtrx = np.zeros((sample_size[0] + 2 * bound, sample_size[1] + 2 * bound), dtype=np.float32)
#     for y in range(0, mtrx.shape[0]):
#         w_y = _get_weight_item(y, sample_size[0], bound)
#         for x in range(0, mtrx.shape[1]):
#             w_x = _get_weight_item(x, sample_size[1], bound)
#             mtrx[y, x] = w_y * w_x
#     return mtrx

# m = _get_weight_mtrx((6, 6), 2)
# print(m)

# def test1():
#     example = np.ones((5, 4))
#     H, W = example.shape
#     out = np.zeros_like(example)
#
#     bound = 2
#     sample_size = (6, 6)
#
#     h, w = sample_size
#     height = h + 2 * bound
#     width = w + 2 * bound
#
#     blocks = []
#     for y in range(-bound, H-bound, h):
#         for x in range(-bound, W-bound, w):
#             rigth_x_bound = max(bound,
#                                 x + width - W)
#             bottom_y_bound = max(bound,
#                                  y + height - H)
#             blocks.append({'x': x,
#                            'y': y,
#                            'height': height,
#                            'width': width,
#                            'bounds':
#                                [[bound, bottom_y_bound], [bound, rigth_x_bound]],
#                            })
#     print(blocks)
#     mtrx = _get_weight_mtrx(sample_size, bound)
#     print(mtrx)
# test1()

