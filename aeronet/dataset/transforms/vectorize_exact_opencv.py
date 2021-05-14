import cv2
import numpy as np
from collections import defaultdict
from rasterio.transform import IDENTITY, xy
import shapely
from shapely.geometry import Polygon, Point
from shapely.affinity import affine_transform
import shapely.ops
def _vectorize_exact_opencv(binary_image, transform=IDENTITY):
    """
    Vectorize binary image, returns a 4-level list of floats [[[[X,Y]]]]

    Args:
        binary_image (numpy.ndarray): image in grayscale HxW or HxWx1
        epsilon (float): approximates a contour shape to another shape with less
            number of vertices depending upon the precision we specify, %
        transform (list of float):  affine transform matrix (first two lines)
            list with 6 params - [GSDx, ROTy, ROTx, GSDy, X, Y]
        upscale (float): scale image for better precision of the polygon. The polygon is correctly downscaled back

    """
    # remove possible 1-sized dimension
    binary_image = np.squeeze(binary_image)
    if binary_image.ndim != 2:
        raise ValueError('Input image must have 2 non-degenerate dimensions')

    # search for all contours
    contours_result = cv2.findContours(
        binary_image,
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # For compatibility with opencv3, where contours_result[0]=image, so we should throw it away
    contours, hierarchy = contours_result if len(contours_result) == 2 else contours_result[1:]

    if not contours:
        return []

    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    geometries = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours:
            assert cnt.shape[1] == 1
            # take a contour as a primary shape
            # take a contour as a primary shape
            shell = get_exact_contour(cnt)
            # add all children as secondary contours (holes)
            holes = [get_hole(c) for c in cnt_children.get(idx, [])]
            poly = shell.difference(shapely.ops.unary_union(holes))
            # switch from pixel positions to projected coordinates
            # affine transform in shapely uses another order or values
                #shapely_tr = [transform[0], transform[1], transform[3],
                #          transform[4], transform[2] + transform[0]/2, transform[5] + transform[4]/2]
                #poly = affine_transform(poly, shapely_tr[:6])

            def affine(x,y,z=None):
                return xy(transform, y, x)
            poly = shapely.ops.transform(affine, poly)

            # append geometry
            geometries.append(shapely.geometry.mapping(poly))
    return geometries


def get_exact_contour(cv_contour):
    cnt = cv_contour.squeeze(1)
    if cnt.shape[0] < 3:
        inside = Polygon()
    else:
        inside = Polygon(cnt).buffer(0)
    bounds = shapely.ops.unary_union([Point(p).buffer(0.5, cap_style=3) for p in cnt])
    res = inside.union(bounds)
    return res


def get_hole(cv_contour):
    cnt = cv_contour.squeeze(1)
    if cnt.shape[0] < 3:
        inside = Polygon()
    else:
        inside = Polygon(cnt).buffer(0)
    bounds = shapely.ops.unary_union([Point(p).buffer(0.5, cap_style=3) for p in cnt])
    res = inside.difference(bounds)
    return res

