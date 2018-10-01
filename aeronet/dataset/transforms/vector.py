from ..vector import FeatureCollection


def merge_dedupe(fcs):
    """Merge feature collections of and remove duplicates"""

    def other(i, fcs):
        fc = FeatureCollection([])
        for j in range(len(fcs)):
            if i != j:
                fc.extend(fcs[j])
        return fc

    def greater(area, areas):
        for a in areas:
            if area < a:
                return False
        return True

    features = []

    for i in range(len(fcs)):

        this_fc = fcs[i]
        other_fc = other(i, fcs)

        for f in this_fc:
            inters_fc = other_fc.intersection(f)

            areas = []
            for ff in inters_fc:
                intersection = f.intersection(ff).area

                if intersection:
                    areas.append(ff.area)

            if len(inters_fc) == 0:
                features.append(f)

            elif greater(f.area, areas):
                features.append(f)

    return FeatureCollection(features, crs=fcs[0].crs)