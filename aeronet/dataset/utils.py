import os
import re
import glob


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

    # construct pattern
    names = '|'.join(names)
    extensions = '|'.join(extensions)
    pattern = '.*({})\.({})$'.format(names, extensions)

    # extract matching file paths
    paths = glob.glob(os.path.join(directory, '*'))
    paths = [path for path in paths if re.match(pattern, path) is not None]

    return paths