import rasterio
import os
from tqdm.notebook import tqdm


def merge_images_and_masks(data_root: str,
                           out_root: str,
                           img_folder: str = 'image',
                           mask_folder: str = 'label',
                           mask_appendix: str = '',
                           downscale: float = 1.):
    img_folder = os.path.join(data_root, img_folder)
    mask_folder = os.path.join(data_root, mask_folder)
    ext = 'tif'
    for idx, file in tqdm(
            enumerate([f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f)) and
                                                            f.split('.')[-1] == ext])):

        with rasterio.open(os.path.join(img_folder, file)) as f:
            rgb = f.read(out_shape=(f.count,
                                    int(f.height * downscale),
                                    int(f.width * downscale)))
            profile = f.profile
            profile['transform'] = f.transform * f.transform.scale((f.width / rgb.shape[-1]),
                                                                   (f.height / rgb.shape[-2]))

        with rasterio.open(os.path.join(mask_folder, file.split('.')[0]+mask_appendix+'.tif')) as f:
            mask = f.read(out_shape=(f.count,
                                     int(f.height * downscale),
                                     int(f.width * downscale)))

        profile['width'], profile['height'] = rgb.shape[-1], rgb.shape[-2]
        profile['nodata'] = 0
        profile['dtype'] = rasterio.uint8
        profile['photometric'] = 'rgb'
        profile['compress'] = 'none'
        profile['count'] = rgb.shape[0] + mask.shape[0]

        with rasterio.open(os.path.join(out_root, str(idx) + '.tif'), 'w', **profile) as dst:
            for ch in rgb.shape[0]:
                dst.write(rgb[ch], ch+1)
            for ch in mask.shape[0]:
                dst.write(mask[ch], rgb.shape[0]+ch+1)
                