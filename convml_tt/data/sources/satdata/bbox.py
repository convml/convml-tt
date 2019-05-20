import itertools

import shapely.geometry as geom
import numpy as np

class LatLonBox():
    def __init__(self, extent):
        """
        extent: [(lon_min, lat_min), (lon_max, lat_max)]
        """
        self.extent = extent

    def get_bounds(self):
        """
        From the extent compute the four corners of the bounding box
        """
        corners = list(itertools.product(*np.array(self.extent).T))
        corners.insert(0, corners.pop(2))

        return corners

    def get_outline_shape(self):
        """return a shapely shape valid for plotting"""

        return geom.Polygon(self.get_bounds())
