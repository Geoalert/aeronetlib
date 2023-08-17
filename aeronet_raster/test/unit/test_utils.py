from rasterio.crs import CRS
from rasterio.transform import Affine
from aeronet_raster.aeronet_raster.utils.coords import get_utm_zone
from aeronet_raster.aeronet_raster.collectionprocessor import CollectionProcessor
from aeronet_raster.aeronet_raster.utils.utils import parse_directory
from aeronet_raster.aeronet_raster.bandcollection.bandcollection import BandCollection


def test_utm_zone_from_latlon():
    assert get_utm_zone(CRS.from_epsg(4326),
                        Affine(0.00001, 0, 1.0, 0, -0.00001, 1.0),
                        (100, 100)) == CRS.from_epsg(32631)
    assert get_utm_zone(CRS.from_epsg(4326),
                        Affine(0.00001, 0, 1.0, 0, -0.00001, -1.0),
                        (100, 100)) == CRS.from_epsg(32731)


def test_utm_zone_from_projected():
    assert get_utm_zone(CRS.from_epsg(3857),
                        Affine(1.0, 0, 1000.0, 0, -1.0, 1000.0),
                        (100, 100)) == CRS.from_epsg(32631)
    assert get_utm_zone(CRS.from_epsg(3857),
                        Affine(1.0, 0, 1000.0, 0, -1.0, -1000.0),
                        (100, 100)) == CRS.from_epsg(32731)


def test_collection_processor():
    def read_bc(path, bands):
        bands = parse_directory(path, bands)
        return BandCollection(bands)

    def processing_fn(sample):
        out = sample.numpy()
        return out

    path = ''
    input_channels = ['RED', 'GRN', 'BLU']
    bc = read_bc(path, input_channels)

    output_labels = ['RED1', 'GRN1', 'BLU1']
    sample_size = (1024, 1024)
    bound = 256
    bound_mode = 'weight'
    cp = CollectionProcessor(
        input_channels=input_channels,
        output_labels=output_labels,
        processing_fn=processing_fn,
        sample_size=sample_size,
        bound=bound,
        verbose=False,
        bound_mode=bound_mode
    )
    labels_bc = cp.process(bc, path)
    assert labels_bc.numpy() == bc.numpy()
    # path = '/home/adeshkin/projects/urban-local-pipeline/data/head_style_transfer_pda_vs_cyclegan/aze_samples/1/input'
    # bc = read_bc(path, ['RED', 'GRN', 'BLU'])
    # predictor.process(bc, path)
    #
    # import rasterio
    # import numpy as np
    #
    # with rasterio.open(f'{path}/BLU1.tif') as f:
    #     img1 = f.read().astype(np.uint8)
    #
    # with rasterio.open(f'{path}/BLU.tif') as f:
    #     img = f.read()
    #     kwargs = f.meta.copy()
    # d = (np.abs(img - img1) > 0)[0]
    # y_inds, x_inds = np.nonzero(d)
    # print(len(y_inds), len(x_inds))
    # print(y_inds[:5], y_inds[-5:])
    # print(x_inds[:5], x_inds[-5:])
    # print(d.shape)
