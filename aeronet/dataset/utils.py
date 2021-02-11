import os
import re
import glob
from warnings import warn


def parse_directory(directory, names, extensions=('tif', 'tiff', 'TIF', 'TIFF')):
    """
    Extract necessary filenames
    Args:
        directory: str
        names: tuple of str, band or file names, e.g. ['RED', '101']
        extensions: tuple of str, allowable file extensions

    Returns:
        list of matched paths
    """
    paths = glob.glob(os.path.join(directory, '*'))
    extensions = '|'.join(extensions)
    res = []
    for name in names:
        # the channel name must be either full filename (that is, ./RED.tif) or a part after '_' (./dse_channel_RED.tif)
        sep = os.sep if os.sep != '\\' else '\\\\'
        pattern = '.*(^|{}|_)({})\.({})$'.format(sep, name, extensions)
        band_path = [path for path in paths if re.match(pattern, path) is not None]

        # Normally with our datasets it will never be the case, and may indicate wrong file naming
        if len(band_path) > 1:
            warn(RuntimeWarning(
                "There are multiple files matching the channel {}. "
                "It can cause ambiguous behavior later.".format(name)))
        res += band_path

    return res

