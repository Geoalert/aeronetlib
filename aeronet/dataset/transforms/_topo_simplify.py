import topojson
import geojson

from ..vector import FeatureCollection, Feature

# ================= topojson-to-geojson conversion, derived from github.com/datmos/python-topojson====

# todo: rewrite it, delete all that is not used properly like convert_<> functions,
#  leaving only contour creation from arcs
# maybe use https://github.com/kylepollina/topo2geo/


TYPEGEOMETRIES = (
    'LineString',
    'MultiLineString',
    'MultiPoint',
    'MultiPolygon',
    'Point',
    'Polygon',
    'GeometryCollection'
)


class Transformer:

    def __init__(self, arcs, transform):
        self.translate = transform['translate']
        self.scale = transform['scale']
        self.arcs = list(map(self.convert_arc, arcs))

    def convert_arc(self, arc):
        out_arc = []
        for point in arc:
            out_arc.append(self.convert_point(point))
        return out_arc

    def reversed_arc(self, arc):
        return list(reversed(self.arcs[~arc]))

    def stitch_arcs(self, arcs):
        line_string = []
        for arc in arcs:
            if arc < 0:
                line = self.reversed_arc(arc)
            else:
                line = self.arcs[arc]
            if len(line_string) > 0:
                if line_string[-1] == line[0]:
                    line_string.extend(line[1:])
                else:
                    line_string.extend(line)
            else:
                line_string.extend(line)
        return line_string

    def stich_multi_arcs(self, arcs):
        return list(map(self.stitch_arcs, arcs))

    def convert_point(self, point):
        return [
            point[0] *
            self.scale[0] +
            self.translate[0],
            point[1] *
            self.scale[1] +
            self.translate[1]]

    def feature(self, feature):
        out = {'type': 'Feature',
               'geometry': {'type': feature['type']}}
        if feature['type'] in ('Point', 'MultiPoint'):
            out['geometry']['coordinates'] = feature['coordinates']
        elif feature['type'] in ('LineString', 'MultiLineString', 'MultiPolygon', 'Polygon'):
            out['geometry']['arcs'] = feature['arcs']
        elif feature['type'] == 'GeometryCollection':
            out['geometry']['geometries'] = feature['geometries']
        for key in ('properties', 'bbox', 'id'):
            if key in feature:
                out[key] = feature[key]
        out['geometry'] = self.geometry(out['geometry'])
        return out

    def geometry(self, geometry):
        if geometry['type'] == 'Point':
            return self.point(geometry)
        elif geometry['type'] == 'MultiPoint':
            return self.multi_point(geometry)
        elif geometry['type'] == 'LineString':
            return self.line_string(geometry)
        elif geometry['type'] == 'MultiLineString':
            return self.multi_line_string_poly(geometry)
        elif geometry['type'] == 'Polygon':
            return self.multi_line_string_poly(geometry)
        elif geometry['type'] == 'MultiPolygon':
            return self.multi_poly(geometry)
        elif geometry['type'] == 'GeometryCollection':
            return self.geometry_collection(geometry)

    def point(self, geometry):
        geometry['coordinates'] = self.convert_point(geometry['coordinates'])
        return geometry

    def multi_point(self, geometry):
        geometry['coordinates'] = list(map(self.convert_point, geometry['coordinates']))
        return geometry

    def line_string(self, geometry):
        geometry['coordinates'] = self.stitch_arcs(geometry['arcs'])
        del geometry['arcs']
        return geometry

    def multi_line_string_poly(self, geometry):
        geometry['coordinates'] = self.stich_multi_arcs(geometry['arcs'])
        del geometry['arcs']
        return geometry

    def multi_poly(self, geometry):
        geometry['coordinates'] = list(map(self.stich_multi_arcs, geometry['arcs']))
        del geometry['arcs']
        return geometry

    def geometry_collection(self, geometry):
        out = {'type': 'FeatureCollection'}
        out['features'] = list(map(self.feature, geometry['geometries']))
        return out


def topo2geo(topo: topojson.Topology):
    """
    Convert topology to geojson object
    :param topo:
    :return:
    """
    topo_dict = topo.to_dict()
    input_name = list(topo_dict['objects'].keys())[0]
    # it seems that in case of topojson.Topology.to_dict() input_name === 'data'. Maybe replace
    geo = topo_dict['objects'][input_name]

    if 'transform' in topo_dict.keys():
        transform = topo_dict['transform']
    else:
        # Transform is used for optimal data compression; it can be not present if the data is not quantized.
        # Actually, in our process we do not quantize the data, so it is the main case, but we leave the transform here
        # for the function to be able to process other topojson data
        transform = {'translate':[0.0, 0.0], 'scale':[1.0, 1.0]}

    transformer = Transformer(topo_dict['arcs'], transform)
    if geo['type'] in TYPEGEOMETRIES:
        geo = transformer.geometry(geo)
    return geo
# =================


def _clean_feature(feat):
    """
    The topojson to geojson transform can produce 2-point features, consisting of one arc.
    We need to delete such features because they have invalid shapes
    :param feat: geojson.Feature
    :return:
    """
    new_poly = []
    for contour in feat['geometry']['coordinates']:
        if len(contour) > 2 and \
                not (len(contour) == 3 and contour[0] == contour[-1]):
            new_poly.append(contour)
    if len(new_poly) != 0:
        return new_poly


def projected_geojson(fc):
    """
    Same as fc.geojson property but return non-standard projected geojson
    Args:
        fc: aeronet FeatureCollection

    Returns:
        geojson.FeatureCollection object
    """
    return geojson.FeatureCollection(crs=fc.crs.to_dict(),
                                     features=[geojson.Feature(geometry=f.geometry,
                                                               properties=f.properties) for f in fc])


def topo_simplify(fc: FeatureCollection, geo_epsilon=0):
    """
    Simplifies a FeatureCollection with Douglas-Peucker (?) algorithm preserving the topology.
    It is needed if the geometries in the fc have joint arcs which can be used for topojson creation.
    If this is the case, these joint edges are preserved and the resulting simplified geometries do not disconnect
    and do not intersect. If the input geometries are disconnected, they are simplified separately, and this function
    works not better than the typical polygonize with simplification
    Args:
        fc: FeatureCollection to be simplified
        geo_epsilon: the allowed simplification error. Is measured in the projection units, so it is in meters in case
            of most projected CRSs and in lat\lon degrees in case of EPSG:4326 crs. It differs from the
            simplification parameter in polygonize, as there it is bound to the pixel size, which is more reasonable.
            So in order to preserve the same behavior, we should use the polygonized sample's resolution
            and pass it here as `geo_epsilon = sample.res*epsilon'
    :return: new FeatureCollection with simplified geometry
    """
    topo_collection = topojson.Topology(projected_geojson(fc),
                                        prequantize=False,
                                        toposimplify=geo_epsilon,
                                        topology=True)

    new_features = []
    new_fc = topo2geo(topo_collection)

    for feat in new_fc['features']:
        new_poly = _clean_feature(feat)
        if new_poly:
            new_features.append(Feature(properties=feat['properties'],
                                        geometry={'type': 'Polygon', 'coordinates': new_poly},
                                        crs=fc.crs))

    new_fc = FeatureCollection(features=new_features, crs=fc.crs)
    return (new_fc)