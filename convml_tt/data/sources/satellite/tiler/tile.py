"""
Utilities to create (approximate) square tiles from lat/lon satelite data
"""
import itertools

import cartopy.crs as ccrs
import luigi
import matplotlib.pyplot as plt
import numpy as np
import satpy
import shapely.geometry as geom
import xesmf

from ...dataset import ImageTripletDataset
from . import grid as tile_grid
from . import regridding as tile_regridding
from . import satpy_rgb
from . import utils as tile_utils
from .utils import create_true_color_img

TILE_FILENAME_FORMAT = ImageTripletDataset.TILE_FILENAME_FORMAT


class RectTile:
    class TileBoundsOutsideOfInputException(Exception):
        pass

    def __init__(self, lat0, lon0, l_meridional, l_zonal):
        """
        Locally Cartesian tile centered on (lat,lon)=(`lat0`, `lon0`)
        with length in the zonal and meridional directions given by `l_zonal`
        and `l_meridional` (in meters)
        """
        self.lat0 = lat0
        self.lon0 = lon0
        self.l_meridional = l_meridional
        self.l_zonal = l_zonal

    def get_bounds(self):
        """
        The the lat/lon bounds of the tile. First calculates the approximate
        lat/lon distance as if the tile was centered on the equator and then
        uses a rotated pole projection to move the title
        """
        ldeg_lon = tile_grid.get_approximate_equator_deg_dist(l_dist=self.l_zonal)
        ldeg_lat = tile_grid.get_approximate_equator_deg_dist(l_dist=self.l_meridional)

        corners_dir = list(itertools.product([1, -1], [1, -1]))
        corners_dir.insert(0, corners_dir.pop(2))

        corners = np.array([ldeg_lon / 2.0, ldeg_lat / 2.0]) * np.array(corners_dir)

        return tile_grid.transform_latlon_from_equator(
            lon=corners[:, 0], lat=corners[:, 1], lat0=self.lat0, lon0=self.lon0
        )

    def get_outline_shape(self):
        """return a shapely shape valid for plotting"""

        return geom.Polygon(self.get_bounds())

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
        return tile_utils.crop_field_to_latlon_box(
            da=da, latlon_pts=self.get_bounds().T, pad_pct=pad_pct
        )

    def resample(
        self, da, resolution, method="bilinear", crop_pad_pct=0.1, keep_attrs=False
    ):
        """
        Resample a xarray DataArray onto this tile with grid made of NxN points

        resolution should either be given as `resolution=dict(dx=1e3)` to set
        the resolution in meters or as `resolution=dict(N=256)` to define the
        number of pixels in the output
        """
        da_cropped = self.crop_field(da=da, pad_pct=crop_pad_pct)

        if da_cropped.x.count() == 0 or da_cropped.y.count() == 0:
            raise self.TileBoundsOutsideOfInputException

        size = dict(xy=(self.l_zonal, self.l_meridional), **resolution)

        return tile_regridding.rect_resample(
            da=da_cropped,
            lat0=self.lat0,
            lon0=self.lon0,
            size=size,
            method=method,
            keep_attrs=keep_attrs,
        )

    def plot_outline(self, ax=None, alpha=0.6, **kwargs):
        if ax is None:
            crs = ccrs.PlateCarree()
            _, ax = plt.subplots(subplot_kw=dict(projection=crs), figsize=(10, 6))
            ax.gridlines(linestyle="--", draw_labels=True)
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


def _rgb_da_to_img(da):
    # need to sort by y otherize resulting image is flipped... there must be a
    # better way
    da_ = da.sortby("y", ascending=False)
    return satpy.writers.get_enhanced_image(da_)


class GenerateSceneTiles(luigi.Task):
    scene_id = luigi.Parameter()
    tile_type = luigi.Parameter()
    tile_size = luigi.Parameter()  # size in [m]
    tile_N = luigi.Parameter()  # number of nx, ny values

    def requires(self):
        tasks = {}
        tasks["tiles_meta"] = GenerateTripletsTilesMeta()
        tasks["scene_rgb"] = CreateRGBScene(scene_id=self.scene_id)
        return tasks

    def run(self):
        tile_size = self.tile_size
        tile_N = self.tile_N
        output_dir = Path("blah")

        da_scene = self.input()["scene_rgb"].open()
        scene_tiles_meta = self.input()["tiles_meta"]

        for tile_meta in scene_tiles_meta:
            tile = RectTile(
                lat0=tile_meta["lat0"],
                lon0=tile_meta["lon0"],
                l_zonal=tile_size,
                l_meridional=tile_size,
            )

            da_tile_rgb = tile.resample(
                da=da_scene, resolution=dict(N=tile_N), keep_attrs=True
            )
            tile_img = satpy_rgb.truecolor_rgb_to_img(da_tile_rgb)

            fn_out = TILE_FILENAME_FORMAT.format(
                tile_id=tile_meta["tile_id"],
                tile_type=tile_meta["tile_type"],
                ext="png",
            )
            tile_img.save(output_dir / fn_out, "PNG")

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
