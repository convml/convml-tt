import itertools

import cartopy.crs as ccrs
import shapely.geometry as geom
import numpy as np
import matplotlib.pyplot as plt


class LatLonBox:
    def __init__(self, bounds):
        """
        bounds: [(lon_min, lat_min), (lon_max, lat_max)]
        """
        self.bounds = bounds

    def get_extent(self, pad=0):
        """
        Return extent compatible with matplotlib.imshow
        [x0 ,x1, y0, y1]
        """
        if pad > 1 or pad == 0:
            x_pad = y_pad = pad
        elif pad != 0:
            x_pad = pad * (self.bounds[1][0] - self.bounds[0][0])
            y_pad = pad * (self.bounds[1][1] - self.bounds[0][1])

        return [
            self.bounds[0][0] - x_pad,
            self.bounds[1][0] + x_pad,
            self.bounds[0][1] - y_pad,
            self.bounds[1][1] + y_pad,
        ]

    def get_bounds(self):
        """
        From the bounds compute the four corners of the bounding box
        """
        corners = list(itertools.product(*np.array(self.bounds).T))
        corners.insert(0, corners.pop(2))

        return corners

    def get_outline_shape(self):
        """return a shapely shape valid for plotting"""

        return geom.Polygon(self.get_bounds())

    def plot_outline(self, ax=None, alpha=0.6, **kwargs):
        if ax is None:
            crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=crs), figsize=(10, 6))
            gl = ax.gridlines(linestyle="--", draw_labels=True)  # noqa
            ax.coastlines(resolution="10m", color="grey")

        ax.add_geometries(
            [
                self.get_outline_shape(),
            ],
            crs=ccrs.PlateCarree(),
            alpha=alpha,
            **kwargs
        )
        return ax
