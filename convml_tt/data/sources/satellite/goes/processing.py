import os
import random
import time
import warnings

import cartopy.crs as ccrs
import numpy as np
import satdata
import xarray as xr
import yaml
from tqdm import tqdm

from . import tiler

TRIPLET_FN_FORMAT = "{:05d}_{}.png"

try:
    from . import satpy_rgb

    HAS_SATPY = True
except ImportError:
    HAS_SATPY = False


def find_datasets_keys(times, dt_max, cli, channels=[1, 2, 3], debug=False):
    """
    query API to find datasets with each required channel
    """

    def get_channel_file(t, channel):
        return cli.query(
            time=t, region="F", debug=debug, channel=channel, dt_max=dt_max
        )

    filenames = []
    for t in times:
        # NOTE: if a key isn't returned for all channels one of the queries
        # will return an empty list which will make zip create an empty list
        # and discard all the channels that are available
        filenames += zip(*[get_channel_file(t=t, channel=c) for c in channels])

    return filenames


class FakeScene(list):
    def __init__(self, source_files, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_files = source_files


def set_projection_attribute_and_scale_coords(ds):
    gp = ds.goes_imager_projection

    globe = ccrs.Globe(
        ellipse="sphere",
        semimajor_axis=gp.semi_major_axis,
        semiminor_axis=gp.semi_minor_axis,
    )
    img_proj = ccrs.Geostationary(
        satellite_height=gp.perspective_point_height,
        central_longitude=gp.longitude_of_projection_origin,
        globe=globe,
    )
    ds.attrs["crs"] = img_proj

    # coordinates are scaled by satellite height in image
    ds.coords["x"] = ds.x * gp.perspective_point_height
    ds.coords["y"] = ds.y * gp.perspective_point_height
    return ds


def _load_channels_old(fns, cli):
    CHUNK_SIZE = 4096  # satpy uses this chunksize, so let's do the same

    def _load_file(fn):
        ds = xr.open_dataset(fn, chunks=dict(x=CHUNK_SIZE, y=CHUNK_SIZE))
        ds = set_projection_attribute_and_scale_coords(ds)

        da = ds.Rad
        da.attrs["crs"] = ds.crs
        da["channel"] = int(cli.parse_key(fn)["channel"])

        return da

    channel_da_arr = [_load_file(fn) for fn in fns]

    # it would be tempting concat these into a single data array here, but we
    # can't because the different channels have different resolution
    da_scene = FakeScene(fns_required, channel_da_arr)

    return da_scene


def load_data_for_rgb(
    datasets_filenames, datasource_cli, bbox_extent, path_composites, use_old=False
):
    REQUIRED_CHANNELS = [1, 2, 3]

    def fn_is_required(fn):
        return int(datasource_cli.parse_key(fn)["channel"]) in REQUIRED_CHANNELS

    if use_old:
        das = []  # dataarrays
        for fns in datasets_filenames:
            fns = list(filter(fn_is_required, fns))
            scene = _load_channels_old(fns=fns, cli=cli)
            das.append(scene)

    else:
        if not HAS_SATPY:
            raise Exception("Must have satpy installed to use new RGB method")

        print("Creating composites in domain bounding box")

        das = []  # dataarrays
        for fns in tqdm(datasets_filenames):
            fns = list(filter(fn_is_required, fns))

            da_rgb_domain = satpy_rgb.get_rgb_composite_in_bbox(
                scene_fns=fns, data_path=path_composites, bbox_extent=bbox_extent
            )
            das.append(da_rgb_domain)

    return das


class ProcessedTile(tiler.Tile):
    @classmethod
    def load(cls, meta_fn):
        tile_id = meta_fn.name.split("_")[0]
        meta = yaml.load(open(meta_fn))
        anchor_meta = meta["target"]["anchor"]
        tile = cls(
            lat0=anchor_meta["lat"], lon0=anchor_meta["lon"], size=anchor_meta["size"]
        )

        setattr(tile, "id", tile_id)
        setattr(tile, "source", meta["target"]["aws_s3_key"])

        return tile

    def get_source(tile, channel_override):
        key_ch1 = tile.source
        key_ch = key_ch1.replace("M3C01", "M3C{:02d}".format(channel_override))

        da_channel = get_file(key_ch)

        return da_channel


if __name__ == "__main__":
    data_path = Path("../data/storage")
    lon = -60  # Barbardos is near -60W

    t_zenith = satdata.calc_nearest_zenith_time_at_loc(-60)
    times = [t_zenith - datetime.timedelta(days=n) for n in range(3, 13)]
