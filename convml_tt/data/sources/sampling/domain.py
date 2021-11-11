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
        Create a new local cartesian domain centered on latlon coordinates
        relative to the origin in the parent domain
        """
        tile_latlon = domain.latlon_from_xy(
            x=domain.x_c - self.x_c,
            y=domain.y_c - self.y_c
        )
        return LocalCartesianSquareTileDomain(
            central_latitude=float(tile_latlon[1]),
            central_longitude=float(tile_latlon[0]),
            size=self.size,
            x_c=self.x_c, y_c=self.y_c,
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

            # TODO: calculating the centre like this is a bit crude since on a
            # curve surface the centreline also curves and so on the northern
            # hemisphere we may want a domain slightly further north to fit it
            # in
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

            edges = dict(
                W=("lon", 0, np.max),
                E=("lon", -1, np.min),
                S=("lat", 0, np.max),
                N=("lat", -1, np.min),
            )

            edge_positions = {}
            for edge, (dim, idx, op) in edges.items():
                ds_edge = ds.isel({dim: idx})
                lat_edge = ds_edge.lat
                lon_edge = ds_edge.lon

                lats_edge, lons_edge = xr.broadcast(lat_edge, lon_edge)

                # find xy-position of the edge
                xy_edge = dummy_domain.crs.transform_points(
                    x=lons_edge.data,
                    y=lats_edge.data,
                    z=np.zeros_like(lats_edge.data),
                    src_crs=ccrs.PlateCarree(),
                )

                if dim == "lon":
                    posn = op(xy_edge[..., 0])
                elif dim == "lat":
                    posn = op(xy_edge[..., 1])
                else:
                    raise NotImplementedError

                edge_positions[edge] = posn

            lx = edge_positions["E"] - edge_positions["W"]
            ly = edge_positions["N"] - edge_positions["S"]

            # round down to nearest meter
            lx = np.round(lx, decimals=0)
            ly = np.round(ly, decimals=0)

            # TODO: remove crop by calculating centre offset which still fit
            # rectangular domain
            crop = 0.95

            # finally we return a domain with the correct span
            return rc.LocalCartesianDomain(
                central_latitude=lat_center,
                central_longitude=lon_center,
                l_zonal=crop * lx,
                l_meridional=crop * ly,
            )
        else:
            raise NotImplementedError(ds.coords)
