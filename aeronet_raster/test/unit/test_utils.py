from rasterio.crs import CRS
from rasterio.transform import Affine
import numpy as np
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


def _run_collection_processor(bc, path, input_channels, output_labels,
                              sample_size, bound, dst_dtype, processing_fn, bound_mode):
    cp = CollectionProcessor(
        input_channels=input_channels,
        output_labels=output_labels,
        processing_fn=processing_fn,
        sample_size=sample_size,
        bound=bound,
        verbose=True,
        bound_mode=bound_mode,
        dst_dtype=dst_dtype
    )

    labels_bc = cp.process(bc, path)

    return labels_bc


def test_collection_processor_bound_mode(get_file):
    path, sample_size, bound, bound_mode, dst_dtype, input_channels, output_labels = get_file

    def read_bc(path, bands):
        bands = parse_directory(path, bands)
        return BandCollection(bands)

    bc = read_bc(path, input_channels)

    proc_fn = lambda sample: sample.numpy()

    labels_bc = _run_collection_processor(bc, path, input_channels, output_labels,
                                          sample_size, bound, dst_dtype, proc_fn, bound_mode)

    # labels_bc should be equal to bc after processing
    assert np.allclose(bc.numpy(), labels_bc.numpy(), atol=2)  # due to rounding can be up to 2
