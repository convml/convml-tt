"""
Utilities to create (approximate) square tiles from lat/lon satelite data
"""
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.patches as mpatches
import shapely.geometry as geom
import warnings
import regridcart as rc


class LocalCartesianSquareTileDomain(rc.LocalCartesianDomain):
    def __init__(self, central_latitude, central_longitude, size, x_c=0.0, y_c=0.0):
        """
        Create a locally Cartesian square tile with `size` (in meters)
        """
        self.size = size
        super().__init__(
            central_latitude=central_latitude,
            central_longitude=central_longitude,
            l_meridional=size,
            l_zonal=size,
            x_c=x_c,
            y_c=y_c,
        )

    def get_grid(self, N):
        dx = self.size / N
        ds_grid = super().get_grid(dx=dx)
        # the floating point devision in `get_grid` when we're passing in `dx`
        # means that sometimes we might produce an extra set of x- or y-values,
        # so we do an index-selection here to ensure those aren't included
        return ds_grid.isel(x=slice(0, N), y=slice(0, N))


class CartesianSquareTileDomain(rc.CartesianDomain):
    def __init__(self, x_c, y_c, size):
        """
        Create a Cartesian square tile with `size` (in meters) centered at a
        (x,y)=(x_c,y_c) location
        """
        self.size = size
        super().__init__(x_c=x_c, y_c=y_c, l_meridional=size, l_zonal=size)

    def get_grid(self, N):
        dx = self.size / N
        ds_grid = super().get_grid(dx=dx)
        # the floating point devision in `get_grid` when we're passing in `dx`
        # means that sometimes we might produce an extra set of x- or y-values,
        # so we do an index-selection here to ensure those aren't included
        return ds_grid.isel(x=slice(0, N), y=slice(0, N))

    def locate_in_latlon_domain(self, domain):
        """
        Using the (x,y) spatial projection of the local cartesian domain `domain`
        """
        tile_latlon = ccrs.PlateCarree().transform_point(
            x=self.x_c, y=self.y_c, src_crs=domain.crs
        )
        return LocalCartesianSquareTileDomain(
            central_latitude=tile_latlon[1],
            central_longitude=tile_latlon[0],
            size=self.size,
            x_c=self.x_c,
            y_c=self.y_c,
        )


class SourceDataDomain:
    """
    Represents that the domain information should be extracted from a source dataset
    """

    def generate_from_dataset(self, ds):
        """
        Create an actual domain instance from the provided dataset. This will
        be the largest possible domain that can fit.
        """
        if "x" in ds.coords and "y" in ds.coords:
            x_min, x_max = ds.x.min().data, ds.x.max().data
            y_min, y_max = ds.y.min().data, ds.y.max().data

            l_zonal = x_max - x_min
            l_meridinonal = y_max - y_min
            x_c = 0.5 * (x_min + x_max)
            y_c = 0.5 * (y_min + y_max)
            return rc.CartesianDomain(
                l_meridional=l_meridinonal, l_zonal=l_zonal, x_c=x_c, y_c=y_c
            )
        elif "lat" in ds.coords and "lon" in ds.coords:
            lat_min, lat_max = ds.lat.min(), ds.lat.max()
            lon_min, lon_max = ds.lon.min(), ds.lon.max()
            lat_center = float((lat_min + lat_max) / 2.0)
            lon_center = float((lon_min + lon_max) / 2.0)
            # create a domain with no span but positioned at the correct point
            # so that ew can calculate the maximum zonal and meridional span
            # from the lat/lon of the corners
            dummy_domain = rc.LocalCartesianDomain(
                central_latitude=lat_center,
                central_longitude=lon_center,
                l_zonal=0.0,
                l_meridional=0.0,
            )

            # lat/lon of corners
            # SW, SE, NW, NE
            lats = np.array([lat_min, lat_min, lat_max, lat_max])
            lons = np.array([lon_min, lon_max, lon_min, lon_max])

            # find xy-position of corners
            xy_corners = dummy_domain.crs.transform_points(
                x=lons, y=lats, z=np.zeros_like(lats), src_crs=ccrs.PlateCarree()
            )
            # use max of W-edge, SW and NW
            x_min = xy_corners[[0, 2], 0].max()
            # use min of E-edge, SE and NE
            x_max = xy_corners[[1, 3], 0].min()
            # use max of S-edge, SW and SE
            y_min = xy_corners[[0, 1], 1].max()
            # use min of N-edge, NW and NE
            y_max = xy_corners[[2, 3], 1].min()

            lx = x_max - x_min
            ly = y_max - y_min

            # round down to nearest meter
            lx = np.round(lx, decimals=0)
            ly = np.round(ly, decimals=0)

            # finally we return a domain with the correct span
            s = 0.99
            return rc.LocalCartesianDomain(
                central_latitude=lat_center,
                central_longitude=lon_center,
                l_zonal=s*lx,
                l_meridional=s*ly,
            )
        else:
            raise NotImplementedError(ds.coords)
