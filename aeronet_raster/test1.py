import numpy as np
from aeronet_raster.get_weight_mtrx import get_weight_mtrx


def test1(img_shape=(5, 4), sample_size=(6, 6), bound=2):
    mtrx = get_weight_mtrx(sample_size, bound)

    example = np.ones(img_shape)
    H, W = example.shape

    h, w = sample_size
    height = h + 2 * bound
    width = w + 2 * bound

    blocks = []
    for y in range(-bound, H-bound, h):
        for x in range(-bound, W-bound, w):
            rigth_x_bound = max(bound,
                                x + width - W)
            bottom_y_bound = max(bound,
                                  y + height - H)
            blocks.append({'x': x,
                           'y': y,
                           'height': height,
                           'width': width,
                           'bounds':
                               [[bound, bottom_y_bound], [bound, rigth_x_bound]],
                           })

    # for block in blocks:
    #     x, y, height, width, bounds = block['x'], block['y'], block['height'], block['width'], block['bounds']
    #     up_bound = bounds[0][0]
    #     bottom_bound = bounds[0][1]
    #     left_bound = bounds[1][0]
    #     right_bound = bounds[1][1]
    #
    #     window_weight_mtrx = np.copy(mtrx)
    #     sample_size = (window_weight_mtrx.shape[0] - 2 * up_bound,
    #                    window_weight_mtrx.shape[1] - 2 * left_bound)
    #
    #     if y < 0:
    #         window_weight_mtrx[:2 * up_bound, 2 * left_bound:sample_size[1]] = 1
    #         for i in range(window_weight_mtrx.shape[0]-2*up_bound, window_weight_mtrx.shape[0]):
    #             for j
    #             window_weight_mtrx[y, :sample_size[1]] = 1
    #     elif x < 0:
    #         window_weight_mtrx[2 * up_bound:sample_size[0], :2 * left_bound] = 1
    #
    #     if y < 0 and x < 0:
    #         window_weight_mtrx[:sample_size[0], :sample_size[1]] = 1
    #
    #
    #
    #     if (((y + sample_size[0]) >= (H - up_bound))
    #             and ((x + sample_size[1]) >= (W - left_bound))):
    #         window_weight_mtrx[sample_size[0]:, sample_size[1]:] = 1
    #     elif (y + sample_size[0]) >= (H - up_bound):
    #         window_weight_mtrx[sample_size[0]:, 2 * left_bound:sample_size[1]] = 1
    #     elif (x + sample_size[1]) >= (W - left_bound):
    #         window_weight_mtrx[2 * up_bound:sample_size[0], sample_size[1]:] = 1
    #
    #     if y < 0 and ((x + sample_size[1]) >= (W - left_bound)):
    #         window_weight_mtrx[:sample_size[0], sample_size[1]:] = 1
    #
    #     if x < 0 and ((y + sample_size[0]) >= (H - up_bound)):
    #         window_weight_mtrx[sample_size[0]:, :sample_size[1]] = 1
    #
    #     if (y + sample_size[0]) >= (H - up_bound):
    #         window_weight_mtrx = window_weight_mtrx[:-bottom_bound]
    #
    #     if (x + sample_size[1]) >= (W - left_bound):
    #         window_weight_mtrx = window_weight_mtrx[:, :-right_bound]
    #
    #     if y < 0:
    #         y += up_bound
    #         window_weight_mtrx = window_weight_mtrx[up_bound:]
    #
    #     if x < 0:
    #         x += left_bound
    #         window_weight_mtrx = window_weight_mtrx[:, left_bound:]
    #
    #
    #
    #     print(window_weight_mtrx)
    #     print(window_weight_mtrx.shape)
    #     #break


if __name__ == '__main__':
    #test1((3, 2))
    m = np.ones((3, 2))
    w1 = m[0, 0]
    w2 = m[0, 1]
    a = (w1 + 2) / (w1 + w2)
    print(a)
