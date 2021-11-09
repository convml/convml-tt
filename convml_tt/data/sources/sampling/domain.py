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
        Create an actual domain instance from the provided dataset
        """
        if "x" in ds.coords and "y" in ds.coords:
            x_min, x_max = ds.x.min().data, ds.x.max().data
            y_min, y_max = ds.y.min().data, ds.y.max().data

            l_zonal = x_max - x_min
            l_meridinonal = y_max - y_min
            x_c = 0.5 * (x_min + x_max)
            y_c = 0.5 * (y_min + y_max)
            return CartesianDomain(
                l_meridional=l_meridinonal, l_zonal=l_zonal, x_c=x_c, y_c=y_c
            )
        elif "lat" in ds.coords and "lon" in ds.coords:
            raise NotImplementedError(LocalCartesianDomain.__name__)
        else:
            raise NotImplementedError(ds.coords)


def deserialise_domain(data):
    if "central_longitude" in data and "central_latitude" in data:
        return LocalCartesianDomain(**data)
    else:
        return CartesianDomain(**data)
