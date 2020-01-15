import itertools

import cartopy.crs as ccrs
import shapely.geometry as geom
import numpy as np
import matplotlib.pyplot as plt

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

    def plot_outline(self, ax=None, alpha=0.6, **kwargs):
        if ax is None:
            crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=crs), figsize=(10, 6))
            gl = ax.gridlines(linestyle='--', draw_labels=True)
            ax.coastlines(resolution='10m', color='grey')

        ax.add_geometries([self.get_outline_shape(),], crs=ccrs.PlateCarree(), alpha=alpha, **kwargs)
        return ax
