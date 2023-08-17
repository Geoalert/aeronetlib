from rasterio.crs import CRS
from rasterio.transform import Affine
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
from aeronet_raster.aeronet_raster.utils.coords import get_utm_zone
from aeronet_raster.aeronet_raster.collectionprocessor import CollectionProcessor
from aeronet_raster.aeronet_raster.utils.utils import parse_directory
from aeronet_raster.aeronet_raster.bandcollection.bandcollection import BandCollection
from generate_files import create_tiff_file


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


def _run_collection_processor(height=3000, width=400, gen_mode='random',
                              sample_size=(1500, 1500), bound=100, bound_mode=None):
    def read_bc(path, bands):
        bands = parse_directory(path, bands)
        return BandCollection(bands)

    def processing_fn(sample):
        out = sample.numpy()
        return out

    input_channels = ['input']
    output_labels = ['output']

    tempdir = TemporaryDirectory()
    path = Path(tempdir.name)
    filename = path / 'input.tif'
    create_tiff_file(filename, width, height, mode=gen_mode, count=1)

    bc = read_bc(path, input_channels)

    if bound_mode is None:
        cp = CollectionProcessor(
            input_channels=input_channels,
            output_labels=output_labels,
            processing_fn=processing_fn,
            sample_size=sample_size,
            bound=bound,
            verbose=True
        )
    else:
        cp = CollectionProcessor(
            input_channels=input_channels,
            output_labels=output_labels,
            processing_fn=processing_fn,
            sample_size=sample_size,
            bound=bound,
            verbose=True,
            bound_mode=bound_mode
        )

    labels_bc = cp.process(bc, path)

    in_img = bc.numpy()
    out_img = labels_bc.numpy()

    if bound_mode == 'drop' or bound_mode is None:
        assert out_img.dtype == 'uint8'
    elif bound_mode == 'weight':
        assert out_img.dtype == 'float32'
    assert in_img.shape == out_img.shape
    assert np.max(np.abs(in_img - out_img)) <= 1  # due to rounding can be up to 1
    shutil.rmtree(path)


def test_collection_processor():
    for height in np.random.randint(1, 5000, 3):
        for width in np.random.randint(1, 5000, 3):
            for sample_size0 in np.random.randint(700, 2500, 3):
                for bound in np.random.randint(50, 500, 3):
                    # only even sample_size and bound
                    if sample_size0 % 2 == 1:
                        sample_size0 += 1
                    if bound % 2 == 1:
                        bound += 1

                    _run_collection_processor(height, width, 'random',
                                              (sample_size0, sample_size0), bound, 'weight')
                    _run_collection_processor(height, width, 'random',
                                              (sample_size0, sample_size0), bound, 'drop')
