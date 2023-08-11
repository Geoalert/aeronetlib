#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
# $ pip install twine
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'aeronet'
DESCRIPTION = 'Deep learning with remote sensing data.'
URL = ''
EMAIL = 'hello@geoalert.com'
AUTHOR = 'Geoalert'
REQUIRES_PYTHON = '>=3.6.0'
#
here = os.path.abspath(os.path.dirname(__file__))

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)
VERSION = about['__version__']

# Load the subpackages versions
about_tmp = {}
with open(os.path.join(here, 'aeronet_raster', 'aeronet_raster', '__version__.py')) as f:
    exec(f.read(), about_tmp)
RASTER_VERSION = about_tmp['__version__']

about_tmp = {}
with open(os.path.join(here, 'aeronet_vector', 'aeronet_vector', '__version__.py')) as f:
    exec(f.read(), about_tmp)
VECTOR_VERSION = about_tmp['__version__']

about_tmp = {}
with open(os.path.join(here, 'aeronet_convert', 'aeronet_convert', '__version__.py')) as f:
    exec(f.read(), about_tmp)
CONVERT_VERSION = about_tmp['__version__']

# The minimum installation is empty
REQUIRED = []

# Optionally, aeronet-vector, aeronet-raster and aeronet-convert (requiring both previous) can be added.
# Thus, installation can be in 3 variants: raster-only, vector-only, raster+vector, raster+vector+conversion
# all the 3rd-party requirements are inherited from sublibs
EXTRAS = {
    'raster': [f'aeronet-raster=={RASTER_VERSION}'],
    'vector': [f'aeronet-vector=={VECTOR_VERSION}'],
    'convert': [f'aeronet-convert=={CONVERT_VERSION}'],
    'all': [f'aeronet-vector=={VECTOR_VERSION}',
            f'aeronet-raster=={RASTER_VERSION}',
            f'aeronet-convert=={CONVERT_VERSION}']  # Actually, alias for convert
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!
long_description = DESCRIPTION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print(s)

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=["aeronet", "aeronet/dataset", "aeronet/converters", "aeronet/dataset/io"],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)