"""
Functions for local-cartesian grid with lat/lon positions
"""
import math
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from scipy.constants import pi


def transform_latlon_from_equator(lon, lat, lat0, lon0):
    """
    Transform (lat, lon) values to be centered on (lat0, lon0)
    """
    p = ccrs.RotatedPole(
        pole_latitude=90 + lat0,
        pole_longitude=lon0,
        central_rotated_longitude=-180.0,
    )

    return ccrs.PlateCarree().transform_points(p, lon, lat)[..., :2]


def get_approximate_equator_deg_dist(l_dist):
    # approximate lat/lon distance
    r_earth = 6371e3  # [m]
    return np.arcsin(l_dist / r_earth) * 180.0 / pi


def make_rect_latlon_grid(size, lat0, lon0):
    """
    Get an xarray Dataset containing a rectangular grid centered on (lat0,
    lon0) with lat/lon grid points and their position in meters

    size can be given in three different ways:

    1. xy-size and resolution dx (both in meters):
        `size=dict(xy=(1e5, 2e5), dx=100.0)`

    2. xy-size (in meters) and number of pixels
        `size=dict(xy=(1e5, 2e5), N=256)`

    3. resolution dx (in meters) and number of pixels
        `size=dict(dx=1e3, N=256)`
    """
    if "xy" in size and "dx" in size:
        l_zonal, l_meridional = size["xy"]
        dx = size["dx"]
        n_zonal = math.ceil(l_zonal / dx)
        n_meridional = math.ceil(l_meridional / dx)
    elif "xy" in size and "N" in size:
        l_zonal, l_meridional = size["xy"]
        n_zonal = n_meridional = size["N"]
    elif "N" in size and "dx" in size:
        n_zonal = n_meridional = size["N"]
        l_zonal = l_meridional = n_zonal*size["dx"]
    else:
        raise NotImplementedError(size)

    ldeg_lon = get_approximate_equator_deg_dist(l_dist=l_zonal)
    ldeg_lat = get_approximate_equator_deg_dist(l_dist=l_meridional)

    lon_eq_ = np.linspace(-ldeg_lon / 2.0, ldeg_lon / 2.0, n_zonal)
    lat_eq_ = np.linspace(-ldeg_lat / 2.0, ldeg_lat / 2.0, n_meridional)
    lon_eq, lat_eq = np.meshgrid(lon_eq_, lat_eq_, indexing="ij")

    pts = transform_latlon_from_equator(lon=lon_eq, lat=lat_eq, lat0=lat0, lon0=lon0)

    x = xr.DataArray(
        np.arange(-l_zonal / 2.0, l_zonal / 2, dx),
        attrs=dict(longname="approx distance from center", units="m"),
        dims=("x",),
    )
    y = xr.DataArray(
        np.arange(-l_meridional / 2.0, l_meridional / 2, dx),
        attrs=dict(longname="approx distance from center", units="m"),
        dims=("y",),
    )

    ds = xr.Dataset(coords=dict(x=x, y=y))

    ds["lon"] = xr.DataArray(
        pts[..., 0],
        dims=("x", "y"),
        coords=dict(x=ds.x, y=ds.y),
        attrs=dict(standard_name="grid_longitude", units="degree"),
    )
    ds["lat"] = xr.DataArray(
        pts[..., 1],
        dims=("x", "y"),
        coords=dict(x=ds.x, y=ds.y),
        attrs=dict(standard_name="grid_latitude", units="degree"),
    )

    return ds
