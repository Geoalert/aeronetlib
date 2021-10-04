import os
import re
import glob
from warnings import warn
from osgeo import ogr, osr


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



# Based on https://pcjericks.github.io/py-gdalogr-cookbook/
def convert_file(input_path, input_driver_name, output_path, output_driver_name, input_espg_crs=None, output_espg_crs="ESPG:4326"):
    """
    Extract necessary filenames
    Args:
        directory: str
        input_path: input path
        input_driver_name: ogr driver name string
        output_path: output_path
        output_driver_name: ogr driver name string
        input_espg_crs: crs fomatted as ESPG:dddd
        output_espg_crs:  crs fomatted as ESPG:dddd

    Returns:
        None
    """

    input_driver = ogr.GetDriverByName(input_driver_name)
    output_driver = ogr.GetDriverByName(output_driver_name)


    # output SpatialReference
    out_crs = osr.SpatialReference()
    out_crs.ImportFromEPSG(int(output_espg_crs[5:]))

    # get the input layer
    inDataSet = input_driver.Open(input_path)
    inLayer = inDataSet.GetLayer()
    
    # input SpatialReference
    if input_espg_crs is not None:
        in_crs = osr.SpatialReference()
        in_crs.ImportFromEPSG(input_espg_crs)
    else:
        in_crs = inLayer.GetSpatialRef()
    
    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(in_crs, out_crs)

    # create the output layer
    out_file = output_path
    if os.path.exists(out_file):
        output_driver.DeleteDataSource(out_file)
    outDataSet = output_driver.CreateDataSource(out_file)
    outLayer = outDataSet.CreateLayer(inLayer.GetName(), out_crs, geom_type=inLayer.GetGeomType())

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = inLayer.GetNextFeature()

    # Save and close the files
    inDataSet = None
