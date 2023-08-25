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
        verbose=False,
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
        #non_pad_bounds = block['non_pad_bounds']
        # if non_pad_bounds is not None:
        #     y_min, y_max, x_min, x_max = non_pad_bounds
        #     non_pad_sample = sample[:, y_min:y_max + 1, x_min:x_max + 1]
        #
        #     top = y_min
        #     left = x_min
        #     bottom = sample.shape[1] - non_pad_sample.shape[1] - top
        #     right = sample.shape[2] - non_pad_sample.shape[2] - left
        #     non_pad_sample = non_pad_sample[0, :, :]
        #     border_sample = cv2.copyMakeBorder(non_pad_sample, top, bottom, left, right, 4)
        #     #sample = np.expand_dims(border_sample, 0)

        yield sample, pad_sample, block


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
    # (path, sample_size, bound, bound_mode, dst_dtype,
    #  input_channels, output_labels, padding, dst_nodata) = get_file_padding

    def read_bc(path, bands):
        bands = parse_directory(path, bands)
        return BandCollection(bands)
    path = '/home/adeshkin/projects/urban-local-pipeline/data/klin/1229362'
    input_channels = ['RED', 'GRN', 'BLU']
    sample_size = (904, 904)
    bound = 300
    padding = 'mirror'
    bc = read_bc(path, input_channels)
    nodata = 0
    i = 0
    for sample, pad_sample, block in _run_seq_sampler(bc, input_channels, sample_size, bound, padding):
        sample = sample.transpose(1, 2, 0)
        pad_sample = pad_sample.transpose(1, 2, 0)

        #assert np.all(sample == pad_sample)
        if sample.max() > 0:
            i += 1
            cv2.imwrite('./sample.png', cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            cv2.imwrite('./pad_sample.png', cv2.cvtColor(pad_sample, cv2.COLOR_RGB2BGR))
            non_pad_bounds = block['non_pad_bounds']
            if non_pad_bounds is not None:
                # y_min, y_max, x_min, x_max = non_pad_bounds
                # pad_sample[:y_min].fill(nodata)
                # pad_sample[y_max + 1:].fill(nodata)
                # pad_sample[:, :x_min].fill(nodata)
                # pad_sample[:, x_max + 1:].fill(nodata)
                pad_sample *= np.expand_dims(non_pad_bounds, 2)

                cv2.imwrite('./back_pad_sample.png', cv2.cvtColor(pad_sample, cv2.COLOR_RGB2BGR))
            assert np.all(sample == pad_sample)
            if i == 2:
                print(block['y'], block['x'])

                break


def test_padding():
    def read_bc(path, bands):
        bands = parse_directory(path, bands)
        return BandCollection(bands)
    path = '/home/adeshkin/projects/urban-local-pipeline/data/klin/1229362'
    input_channels = ['RED', 'GRN', 'BLU']
    output_labels = ['r', 'g', 'b']
    sample_size = (904, 904)
    bound = 300
    bound_mode = 'drop'
    padding = 'mirror'
    dtype = 'uint8'
    bc = read_bc(path, input_channels)

    cp = CollectionProcessor(
        input_channels=input_channels,
        output_labels=output_labels,
        processing_fn=lambda x: x,
        sample_size=sample_size,
        bound=bound,
        verbose=False,
        bound_mode=bound_mode,
        padding=padding,
        dtype=dtype
    )

    labels_bc = cp.process(bc, path)

