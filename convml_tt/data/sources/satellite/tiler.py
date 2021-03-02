"""
Utilities to create (approximate) square tiles from lat/lon satelite data
"""
import cartopy.crs as ccrs
import xesmf
import xarray as xr
import numpy as np
from scipy.constants import pi
import shapely.geometry as geom
from pathlib import Path
import matplotlib.pyplot as plt

import itertools
import warnings

from .utils import create_true_color_img

import os

from xesmf.backend import esmf_grid, add_corner, esmf_regrid_build, esmf_regrid_finalize

import tempfile
import math

try:
    from . import satpy_rgb

    HAS_SATPY = True
except ImportError:
    HAS_SATPY = False


class SilentRegridder(xesmf.Regridder):
    def _write_weight_file(self):
        if os.path.exists(self.filename):
            if self.reuse_weights:
                return  # do not compute it again, just read it
            else:
                os.remove(self.filename)

        regrid = esmf_regrid_build(
            self._grid_in, self._grid_out, self.method, filename=self.filename
        )
        esmf_regrid_finalize(regrid)  # only need weights, not regrid object


def crop_field_to_latlon_box(da, box, pad_pct=0.1):
    xs, ys, _ = da.crs.transform_points(ccrs.PlateCarree(), *box).T

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    lx = x_max - x_min
    ly = y_max - y_min

    x_min -= pad_pct * lx
    y_min -= pad_pct * ly
    x_max += pad_pct * lx
    y_max += pad_pct * ly

    if da.x[0] > da.x[-1]:
        x_slice = slice(x_max, x_min)
    else:
        x_slice = slice(x_min, x_max)

    if da.y[0] > da.y[-1]:
        y_slice = slice(y_max, y_min)
    else:
        y_slice = slice(y_min, y_max)

    return da.sel(x=x_slice, y=y_slice)


class RectTile:
    class TileBoundsOutsideOfInputException(Exception):
        pass

    def __init__(self, lat0, lon0, l_meridional, l_zonal):
        self.lat0 = lat0
        self.lon0 = lon0
        self.l_meridional = l_meridional
        self.l_zonal = l_zonal

        regridder_basedir = Path("/nfs/a289/earlcd/tmp/regridding")
        if not regridder_basedir.exists():
            regridder_basedir = Path("/tmp/regridding")
        regridder_basedir.mkdir(exist_ok=True, parents=True)
        self.regridder_tmpdir = Path(tempfile.mkdtemp(dir=regridder_basedir))

    def get_bounds(self):
        """
        The the lat/lon bounds of the tile. First calculates the approximate
        lat/lon distance as if the tile was centered on the equator and then
        uses a rotated pole projection to move the title
        """
        ldeg_lon = self._get_approximate_equator_deg_dist(l=self.l_zonal)
        ldeg_lat = self._get_approximate_equator_deg_dist(l=self.l_meridional)

        corners_dir = list(itertools.product([1, -1], [1, -1]))
        corners_dir.insert(0, corners_dir.pop(2))

        corners = np.array([ldeg_lon / 2.0, ldeg_lat / 2.0]) * np.array(corners_dir)

        return self._transform_from_equator(lon=corners[:, 0], lat=corners[:, 1])

    def _transform_from_equator(self, lon, lat):
        p = ccrs.RotatedPole(
            pole_latitude=90 + self.lat0,
            pole_longitude=self.lon0,
            central_rotated_longitude=-180.0,
        )

        return ccrs.PlateCarree().transform_points(p, lon, lat)[..., :2]

    @staticmethod
    def _get_approximate_equator_deg_dist(l):
        # approximate lat/lon distance
        r = 6371e3  # [m]
        return np.arcsin(l / r) * 180.0 / 3.14

    def get_outline_shape(self):
        """return a shapely shape valid for plotting"""

        return geom.Polygon(self.get_bounds())

    def get_grid(self, dx):
        """
        Get an xarray Dataset containing the new lat/lon grid points with their
        position in meters
        """
        N_zonal = math.ceil(self.l_zonal / dx)
        N_meridional = math.ceil(self.l_meridional / dx)
        ldeg_lon = self._get_approximate_equator_deg_dist(l=self.l_zonal)
        ldeg_lat = self._get_approximate_equator_deg_dist(l=self.l_meridional)

        lon_eq_ = np.linspace(-ldeg_lon / 2.0, ldeg_lon / 2.0, N_zonal)
        lat_eq_ = np.linspace(-ldeg_lat / 2.0, ldeg_lat / 2.0, N_meridional)
        lon_eq, lat_eq = np.meshgrid(lon_eq_, lat_eq_, indexing="ij")

        pts = self._transform_from_equator(lon=lon_eq, lat=lat_eq)

        x = xr.DataArray(
            np.arange(-self.l_zonal / 2.0, self.l_zonal / 2, dx),
            attrs=dict(longname="approx distance from center", units="m"),
            dims=("x",),
        )
        y = xr.DataArray(
            np.arange(-self.l_meridional / 2.0, self.l_meridional / 2, dx),
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

    def get_grid_extent(self):
        """
        Return grid extent compatible with matplotlib.imshow
        [x0 ,x1, y0, y1] in Cartesian coordinates
        """
        return [
            -self.l_zonal / 2.0,
            self.l_zonal / 2.0,
            -self.l_meridional / 2.0,
            self.l_meridional / 2.0,
        ]

    def crop_field(self, da, pad_pct=0.1):
        return crop_field_to_latlon_box(da=da, box=self.get_bounds().T, pad_pct=pad_pct)

    def resample(self, da, dx, method="bilinear", crop_pad_pct=0.1, keep_attrs=False):
        """
        Resample a xarray DataArray onto this tile with grid made of NxN points
        """
        da_cropped = self.crop_field(da=da, pad_pct=crop_pad_pct)

        if da_cropped.x.count() == 0 or da_cropped.y.count() == 0:
            raise self.TileBoundsOutsideOfInputException

        old_grid = xr.Dataset(coords=da_cropped.coords)

        if not hasattr(da_cropped, "crs"):
            raise Exception(
                "The provided DataArray doesn't have a "
                "projection provided. Please set the `crs` "
                "attribute to contain a cartopy projection"
            )

        latlon_old = ccrs.PlateCarree().transform_points(
            da_cropped.crs,
            *np.meshgrid(da_cropped.x.values, da_cropped.y.values),
        )[:, :, :2]

        old_grid["lat"] = (("y", "x"), latlon_old[..., 1])
        old_grid["lon"] = (("y", "x"), latlon_old[..., 0])

        new_grid = self.get_grid(dx=dx)

        Nx_in, Ny_in = da_cropped.x.shape[0], da_cropped.y.shape[0]
        Nx_out, Ny_out = int(new_grid.x.count()), int(new_grid.y.count())

        regridder_weights_fn = (
            "{method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}"
            "__{lat0}_{lon0}.nc".format(
                lon0=self.lon0,
                lat0=self.lat0,
                method=method,
                Ny_in=Ny_in,
                Nx_in=Nx_in,
                Nx_out=Nx_out,
                Ny_out=Ny_out,
            )
        )

        regridder_weights_fn = str(self.regridder_tmpdir / regridder_weights_fn)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                regridder = SilentRegridder(
                    filename=regridder_weights_fn,
                    reuse_weights=True,
                    ds_in=old_grid,
                    ds_out=new_grid,
                    method=method,
                )
            except ValueError:
                raise self.TileBoundsOutsideOfInputException(
                    "something went wrong" " during regridding :("
                )

        da_resampled = regridder(da_cropped)

        da_resampled["x"] = new_grid.x
        da_resampled["y"] = new_grid.y

        if keep_attrs:
            da_resampled.attrs.update(da.attrs)

        return da_resampled

    def get_pyresample_area_def(self, N):
        """
        When using satpy scenes we're better off using pyresample instead of
        xesmf since it appears faster (I think because it uses dask)
        """
        from pyresample import geometry

        L = self.size
        area_id = "tile"
        description = "Tile local cartesian grid"
        proj_id = "ease_tile"
        x_size = N
        y_size = N
        area_extent = (-L, -L, L, L)
        proj_dict = {
            "a": 6371228.0,
            "units": "m",
            "proj": "laea",
            "lon_0": self.lon0,
            "lat_0": self.lat0,
        }
        area_def = geometry.AreaDefinition(
            area_id, description, proj_id, proj_dict, x_size, y_size, area_extent
        )

        return area_def

    def create_true_color_img(self, da_scene, resampling_dx):
        if isinstance(da_scene, list):
            das_channels_resampled = [
                self.resample(da, dx=resampling_dx) for da in da_scene
            ]
            return create_true_color_img(das_channels=das_channels_resampled)
        else:
            if not HAS_SATPY:
                raise Exception(
                    "Must have satpy installed to be able to "
                    "RGB composites with satpy"
                )

            da_tile_rgb = self.resample(da=da_scene, dx=resampling_dx, keep_attrs=True)

            return satpy_rgb.rgb_da_to_img(da_tile_rgb)

    def serialize_props(self):
        return dict(
            lon=float(self.lon0),
            lat=float(self.lat0),
            size=float(self.size),
        )

    def plot_outline(self, ax=None, alpha=0.6, **kwargs):
        if ax is None:
            crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=crs), figsize=(10, 6))
            gl = ax.gridlines(linestyle="--", draw_labels=True)
            ax.coastlines(resolution="10m", color="grey")

        ax.add_geometries(
            [
                self.get_outline_shape(),
            ],
            crs=ccrs.PlateCarree(),
            alpha=alpha,
            **kwargs,
        )
        return ax


class Tile(RectTile):
    def __init__(self, lat0, lon0, size):
        self.size = size
        super().__init__(lat0=lat0, lon0=lon0, l_meridional=size, l_zonal=size)

    def get_grid(self, N):
        dx = self.size / N
        return super().get_grid(dx=dx)


def triplet_generator(
    da_target_scene,
    tile_size,
    tiling_bbox,
    tile_N,
    da_distant_scene=None,
    neigh_dist_scaling=1.0,
    distant_dist_scaling=10.0,
):
    # generate (lat, lon) locations inside tiling_box

    def _est_tile_size_deg(loc):
        _, lat0 = loc
        R = 6371e3  # Earth's radius in m
        tile_size_deg = np.rad2deg(tile_size / (R * np.cos(np.deg2rad(lat0))))
        return tile_size_deg

    def _point_valid(lon, lat):
        h_ts = 0.5 * _est_tile_size_deg(loc=(lon, lat))

        (lon_min, lat_min), (lon_max, lat_max) = tiling_bbox
        try:
            assert lon_min + h_ts <= lon <= lon_max - h_ts
            assert lat_min + h_ts <= lat <= lat_max - h_ts
            return True
        except:
            return False

    def _generate_latlon():
        (lon_min, lat_min), (lon_max, lat_max) = tiling_bbox

        lat = lat_min + (lat_max - lat_min) * np.random.random()
        lon = lon_min + (lon_max - lon_min) * np.random.random()

        if not _point_valid(lon, lat):
            return _generate_latlon()
        else:
            return (lon, lat)

    def _perturb_loc(loc, scaling):
        theta = 2 * pi * np.random.random()

        tile_size_deg = _est_tile_size_deg(loc=loc)

        r = scaling * tile_size_deg * np.random.normal(loc=1.0, scale=0.1)

        dlon = r * np.cos(theta)
        dlat = r * np.sin(theta)

        new_loc = (loc[0] + dlon, loc[1] + dlat)
        if _point_valid(lon=new_loc[0], lat=new_loc[1]):
            return new_loc
        else:
            return _perturb_loc(loc=loc, scaling=scaling)

    anchor_loc = _generate_latlon()
    neighbor_loc = _perturb_loc(anchor_loc, scaling=neigh_dist_scaling)

    if da_distant_scene is None:
        while True:
            dist_loc = _perturb_loc(anchor_loc, scaling=distant_dist_scaling)
            if _point_valid(dist_loc):
                break
    else:
        dist_loc = _generate_latlon()

    locs = [anchor_loc, neighbor_loc, dist_loc]

    tiles = [Tile(lat0=lat, lon0=lon, size=tile_size) for (lon, lat) in locs]

    # create a list of the three scenes used for creating triplets
    da_scene_set = [da_target_scene, da_target_scene]
    if da_distant_scene is None:
        da_scene_set.append(da_target_scene)
    else:
        da_scene_set.append(da_distant_scene)

    # on each of the three scenes use the three tiles to create a resampled
    # image
    try:
        return [
            (tile, tile.create_true_color_img(da_scene, resampling_N=tile_N))
            for (tile, da_scene) in zip(tiles, da_scene_set)
        ]
    except Tile.TileBoundsOutsideOfInputException:
        return triplet_generator(
            da_target_scene,
            tile_size,
            tiling_bbox,
            tile_N,
            da_distant_scene,
            neigh_dist_scaling,
            distant_dist_scaling,
        )
