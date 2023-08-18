import numpy as np


def calc_weight_item(x, sample_size_x, bound):
    """
        Calculate weight depending on x coordinate
    """
    if x < 2 * bound:
        w_x = x / (2 * bound)
    elif 2 * bound <= x < sample_size_x:
        w_x = 1
    else:
        w_x = (sample_size_x + 2 * bound - x) / (2 * bound)

    return w_x


def calc_weight_mtrx(sample_size, bound):
    """
        Calculate weight matrix
    """
    mtrx = np.zeros((sample_size[0] + 2 * bound, sample_size[1] + 2 * bound),
                    dtype=np.float32)
    # TODO: optimize
    for y in range(0, mtrx.shape[0]):
        w_y = calc_weight_item(y, sample_size[0], bound)
        for x in range(0, mtrx.shape[1]):
            w_x = calc_weight_item(x, sample_size[1], bound)
            mtrx[y, x] = w_y * w_x

    return mtrx


def recalc_up_bound_weight_mtrx(mtrx, bound, sample_size, eps=0):
    """
        Recalculate weight matrix for windows near up bound of the image
        because only 2 windows intersect (in normal case there are 4)
    """
    for i in range(bound, 2 * bound):
        for j in range(2 * bound):
            w1 = mtrx[i, j]
            w2 = mtrx[i, j + sample_size[1]]
            # before: w1 + w2 = 1 - r
            # after: (w1 + k1) + (w2 + k2) = 1, where k1/k2 = w1/w2
            r = 1 - (w1 + w2)
            k1 = (w1 * r) / (w1 + w2 + eps)
            mtrx[i, j] += k1  # w1 + k1
            mtrx[i, j + sample_size[1]] = 1 - mtrx[i, j]  # (w2 + k2) = 1 - (w1 + k1)

    # outside the intersection of windows
    mtrx[:2 * bound, 2 * bound:sample_size[1]].fill(1)

    return mtrx


def recalc_bottom_bound_weight_mtrx(mtrx, bound, sample_size, eps=0):
    """
        Recalculate weight matrix for windows near bottom bound of the image
        because only 2 windows intersect (in normal case there are 4)
    """
    for i in range(sample_size[0], mtrx.shape[0]-bound):
        for j in range(2 * bound):
            w1 = mtrx[i, j]
            w2 = mtrx[i, j + sample_size[1]]
            # before: w1 + w2 = 1 - r
            # after: (w1 + k1) + (w2 + k2) = 1, where k1/k2 = w1/w2
            r = 1 - (w1 + w2)
            k1 = (w1 * r) / (w1 + w2 + eps)
            mtrx[i, j] += k1  # w1 + k1
            mtrx[i, j + sample_size[1]] = 1 - mtrx[i, j]  # (w2 + k2) = 1 - (w1 + k1)

    # outside the intersection of windows
    mtrx[sample_size[0]:, 2 * bound:sample_size[1]].fill(1)

    return mtrx


def recalc_left_bound_weight_mtrx(mtrx, src_mtrx, bound, sample_size, after_flag, eps=0):
    """
        Recalculate weight matrix for windows near left bound of the image
        because only 2 windows intersect (in normal case there are 4)
    """
    if after_flag == 'upbottom':
        return mtrx

    for i in range(2*bound):
        for j in range(bound, 2 * bound):
            w1 = src_mtrx[i, j]
            w2 = src_mtrx[i+sample_size[0], j]
            # before: w1 + w2 = 1 - r
            # after: (w1 + k1) + (w2 + k2) = 1, where k1/k2 = w1/w2
            r = 1 - (w1 + w2)
            k1 = (w1 * r) / (w1 + w2 + eps)
            src_mtrx[i, j] += k1  # w1 + k1
            src_mtrx[i+sample_size[0], j] = 1 - src_mtrx[i, j]  # (w2 + k2) = 1 - (w1 + k1)
            if after_flag == 'up':
                mtrx[i + sample_size[0], j] = src_mtrx[i+sample_size[0], j]
            elif after_flag == 'bottom':
                mtrx[i, j] = src_mtrx[i, j]
            else:
                mtrx[i, j] = src_mtrx[i, j]
                mtrx[i + sample_size[0], j] = src_mtrx[i + sample_size[0], j]

    # outside the intersection of windows
    mtrx[2 * bound:sample_size[0], :2 * bound].fill(1)

    return mtrx


def recalc_right_bound_weight_mtrx(mtrx, src_mtrx, bound, sample_size, after_flag, eps=0):
    """
        Recalculate weight matrix for windows near right bound of the image
        because only 2 windows intersect (in normal case there are 4)
    """
    if after_flag == 'upbottom':
        return mtrx

    for i in range(2 * bound):
        for j in range(sample_size[1], mtrx.shape[1]-bound):
            w1 = src_mtrx[i, j]
            w2 = src_mtrx[i + sample_size[0], j]
            # before: w1 + w2 = 1 - r
            # after: (w1 + k1) + (w2 + k2) = 1, where k1/k2 = w1/w2
            r = 1 - (w1 + w2)
            k1 = (w1 * r) / (w1 + w2 + eps)
            src_mtrx[i, j] += k1  # w1 + k1
            src_mtrx[i + sample_size[0], j] = 1 - src_mtrx[i, j]  # (w2 + k2) = 1 - (w1 + k1)
            if after_flag == 'up':
                mtrx[i + sample_size[0], j] = src_mtrx[i+sample_size[0], j]
            elif after_flag == 'bottom':
                mtrx[i, j] = src_mtrx[i, j]
            else:
                mtrx[i, j] = src_mtrx[i, j]
                mtrx[i + sample_size[0], j] = src_mtrx[i + sample_size[0], j]

    # outside the intersection of windows
    mtrx[2 * bound:sample_size[0], sample_size[1]:].fill(1)

    return mtrx


def recalc_bound_weight_mtrx(y, x, up_bound, left_bound, sample_size, src_weight_mtrx, dst_height, dst_width):
    window_weight_mtrx = np.copy(src_weight_mtrx)
    after_flag = ''

    # up
    if y < 0:
        window_weight_mtrx = recalc_up_bound_weight_mtrx(window_weight_mtrx, up_bound, sample_size)
        after_flag += 'up'

    # bottom
    if (y + sample_size[0] + up_bound) >= dst_height:
        window_weight_mtrx = recalc_bottom_bound_weight_mtrx(window_weight_mtrx, up_bound, sample_size)
        after_flag += 'bottom'

    # left
    if x < 0:
        window_weight_mtrx = recalc_left_bound_weight_mtrx(window_weight_mtrx, np.copy(src_weight_mtrx),
                                                           up_bound, sample_size, after_flag)
    # right
    if (x + sample_size[1] + left_bound) >= dst_width:
        window_weight_mtrx = recalc_right_bound_weight_mtrx(window_weight_mtrx, np.copy(src_weight_mtrx),
                                                            up_bound, sample_size, after_flag)

    if y < 0 and x < 0:
        window_weight_mtrx[:sample_size[0], :sample_size[1]].fill(1)

    if y < 0 and ((x + sample_size[1] + left_bound) >= dst_width):
        window_weight_mtrx[:sample_size[0], sample_size[1]:].fill(1)

    if x < 0 and ((y + sample_size[0] + up_bound) >= dst_height):
        window_weight_mtrx[sample_size[0]:, :sample_size[1]].fill(1)

    if (((y + sample_size[0] + up_bound) >= dst_height)
            and ((x + sample_size[1] + left_bound) >= dst_width)):
        window_weight_mtrx[sample_size[0]:, sample_size[1]:].fill(1)

    return window_weight_mtrx
