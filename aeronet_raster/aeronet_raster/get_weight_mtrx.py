import numpy as np


def get_weight_item(x, window, bound):
    if x < 2 * bound:
        w_x = x / (2 * bound)
    elif 2 * bound <= x < window:
        w_x = 1
    else:  # i >= window:
        w_x = (window + 2 * bound - x) / (2 * bound)

    return w_x


def get_weight_mtrx(sample_size, bound):
    mtrx = np.zeros((sample_size[0] + 2 * bound, sample_size[1] + 2 * bound), dtype=np.float32)
    for y in range(0, mtrx.shape[0]):
        w_y = get_weight_item(y, sample_size[0], bound)
        for x in range(0, mtrx.shape[1]):
            w_x = get_weight_item(x, sample_size[1], bound)
            mtrx[y, x] = w_y * w_x
    return mtrx


def recalc_up_bound_weight_mtrx(mtrx, up_bound, left_bound, sample_size):
    for i in range(2 * up_bound):
        for j in range(2 * left_bound):
            w1 = mtrx[i, j]
            w2 = mtrx[i, j + sample_size[1]]
            r = 1 - (w1 + w2)
            k1 = (w1 * r) / (w1 + w2 + 1e-12)
            mtrx[i, j] += k1
            mtrx[i, j + sample_size[1]] = 1 - mtrx[i, j]

    mtrx[:2 * up_bound, 2 * left_bound:sample_size[1]] = 1

    return mtrx


def recalc_bottom_bound_weight_mtrx(mtrx, up_bound, left_bound, sample_size):
    for i in range(sample_size[0], mtrx.shape[0]):
        for j in range(2 * left_bound):
            w1 = mtrx[i, j]
            w2 = mtrx[i, j + sample_size[1]]
            r = 1 - (w1 + w2)
            k1 = (w1 * r) / (w1 + w2 + 1e-12)
            mtrx[i, j] += k1
            mtrx[i, j + sample_size[1]] = 1 - mtrx[i, j]

    mtrx[sample_size[0]:, 2 * left_bound:sample_size[1]] = 1

    return mtrx


def recalc_left_bound_weight_mtrx(mtrx, up_bound, left_bound, sample_size):
    for i in range(2 * up_bound):
        for j in range(2 * left_bound):
            w1 = mtrx[i, j]
            w2 = mtrx[i+sample_size[0], j]
            r = 1 - (w1 + w2)
            k1 = (w1 * r) / (w1 + w2 + 1e-12)
            mtrx[i, j] += k1
            mtrx[i+sample_size[0], j] = 1 - mtrx[i, j]

    mtrx[2 * up_bound:sample_size[0], :2 * left_bound] = 1

    return mtrx


def recalc_right_bound_weight_mtrx(mtrx, up_bound, left_bound, sample_size):
    for i in range(2 * up_bound):
        for j in range(sample_size[1], mtrx.shape[1]):
            w1 = mtrx[i, j]
            w2 = mtrx[i + sample_size[0], j]
            r = 1 - (w1 + w2)
            k1 = (w1 * r) / (w1 + w2 + 1e-12)
            mtrx[i, j] += k1
            mtrx[i + sample_size[0], j] = 1 - mtrx[i, j]

    mtrx[2 * up_bound:sample_size[0], sample_size[1]:] = 1

    return mtrx


# mtrx = get_weight_mtrx((6, 6), 2)
# # sample_size = (6, 6)
# # up_bound = 2
# # left_bound = 2
# # #print(mtrx)
# # # y < 0
#
#
# print(mtrx)
# # print(mtrx[3, 6])
# # print(mtrx[3, 0])