import numpy as np
from typing import Optional, Final


COLORS: Final[tuple] = ((255, 0, 0),
                        (0, 255, 0),
                        (0, 0, 255),
                        (255, 255, 0),
                        (255, 0, 255),
                        (0, 255, 255))


def _random_color():
    return tuple(np.random.randint(0, 256, 3))


def add_mask(image: np.ndarray,
             mask: np.ndarray,
             colormap: Optional[tuple] = None,
             intensity: float = 0.5):
    """
    Put a mask on the image
    Args:
        image: Image as ndarray (width, height, channels=3),
        mask: Mask as ndarray (width, height) or (width, height, channels),
        colormap: Color for each mask channel, a list of (R, G, B) tuples
        intensity: Mask intensity within [0, 1]:
    """
    assert image.ndim == 3 and image.shape[2] == 3, "Image as ndarray (width, height, channels=3) expected"
    if mask.ndim < 3:
        mask = np.expand_dims(mask, 2)
    assert mask.ndim == 3, "Mask as ndarray (width, height) or (width, height, channels) expected"
    assert image.shape[:2] == mask.shape[:2], 'Shapes mismatch'

    if not colormap:
        colormap = list(COLORS)
    while len(colormap) < mask.shape[2]:
        colormap.append(_random_color())
    image = image.astype(np.uint16)
    rgb_mask = np.zeros((*mask.shape[:2], 3)).astype(np.int16)
    for ch in range(mask.shape[2]):
        rgb_mask += np.stack((mask[:, :, ch]*colormap[ch][0],
                              mask[:, :, ch]*colormap[ch][1],
                              mask[:, :, ch]*colormap[ch][2]), axis=-1)
    image += (rgb_mask*intensity).astype(np.uint16)
    return np.clip(image, 0, 255).astype(np.uint8)
