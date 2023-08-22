from rasterio.crs import CRS
from rasterio.transform import Affine
import numpy as np
import cv2
from aeronet_raster.aeronet_raster.utils.coords import get_utm_zone
from aeronet_raster.aeronet_raster.collectionprocessor import CollectionProcessor, SequentialSampler
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
                              sample_size, bound, dst_dtype, processing_fn,
                              bound_mode, padding, src_nodata):
    cp = CollectionProcessor(
        input_channels=input_channels,
        output_labels=output_labels,
        processing_fn=processing_fn,
        sample_size=sample_size,
        src_nodata=src_nodata,
        bound=bound,
        verbose=True,
        bound_mode=bound_mode,
        padding=padding,
        dst_dtype=dst_dtype
    )

    labels_bc = cp.process(bc, path)

    return labels_bc


def _run_seq_sampler(bc, input_channels, sample_size, bound, padding):
    pad_src = SequentialSampler(bc, input_channels, sample_size, bound, padding)
    for pad_sample, block in pad_src:
        sample = (bc
                  .ordered(*input_channels)
                  .sample(block['y'], block['x'], block['height'], block['width'])
                  .numpy())
        non_pad_bounds = block['non_pad_bounds']
        if non_pad_bounds is not None:
            y_min, y_max, x_min, x_max = non_pad_bounds
            non_pad_sample = sample[:, y_min:y_max + 1, x_min:x_max + 1]

            top = y_min
            left = x_min
            bottom = sample.shape[1] - non_pad_sample.shape[1] - top
            right = sample.shape[2] - non_pad_sample.shape[2] - left
            non_pad_sample = non_pad_sample[0, :, :]
            border_sample = cv2.copyMakeBorder(non_pad_sample, top, bottom, left, right, 4)
            sample = np.expand_dims(border_sample, 0)

        yield sample, pad_sample


def test_collection_processor_bound_mode(get_file):
    path, sample_size, bound, bound_mode, dst_dtype, input_channels, output_labels, padding, src_nodata = get_file

    def read_bc(path, bands):
        bands = parse_directory(path, bands)
        return BandCollection(bands)

    bc = read_bc(path, input_channels)

    proc_fn = lambda sample: sample

    labels_bc = _run_collection_processor(bc, path, input_channels, output_labels,
                                          sample_size, bound, dst_dtype, proc_fn, bound_mode, padding, src_nodata)

    # labels_bc should be equal to bc after processing
    assert np.allclose(bc.numpy(), labels_bc.numpy(), atol=2)  # due to rounding can be up to 2


def test_seq_sampler_padding(get_file_padding):
    (path, sample_size, bound, bound_mode, dst_dtype,
     input_channels, output_labels, padding, dst_nodata) = get_file_padding

    def read_bc(path, bands):
        bands = parse_directory(path, bands)
        return BandCollection(bands)

    bc = read_bc(path, input_channels)

    for sample, pad_sample in _run_seq_sampler(bc, input_channels, sample_size, bound, padding):
        assert np.all(sample == pad_sample)
