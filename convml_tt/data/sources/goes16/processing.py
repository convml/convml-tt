import os
import random

from tqdm import tqdm
import xarray as xr
import cartopy.crs as ccrs

import warnings
import numpy as np
import time
import yaml

from . import tiler
import satdata

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


import random


def pick_one_time_per_date_for_study(
    datasets_filenames, datasource_cli, ensure_each_day_has_training_data=False
):
    dataset_files_by_date = {}

    for fns in datasets_filenames:
        date = datasource_cli.parse_key(str(fns[0]), parse_times=True)[
            "start_time"
        ].date()
        dataset_files_by_date.setdefault(date, []).append(fns)

    def _split_date(datasets_filenames):
        l = list(datasets_filenames)
        if ensure_each_day_has_training_data and len(l) < 2:
            raise Exception(
                "There is only one dataset for the given date "
                "(`{}`), is this a mistake?".format(l[0][0])
            )
        random.shuffle(l)
        return l[:1], l[1:]

    datasets_study = []
    datasets_train = []
    for d in dataset_files_by_date.keys():
        l_study_d, l_train_d = _split_date(dataset_files_by_date[d])
        datasets_study += l_study_d
        datasets_train += l_train_d

    return dict(train=datasets_train, study=datasets_study)


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


TRIPLET_FN_FORMAT = "{:05d}_{}.png"


def generate_tile_triplets(
    scenes,
    tiling_bbox,
    tile_N,
    tile_size,
    output_dir,
    N_triplets,
    max_workers=4,
    neighbor_distant_frac=0.8,
    N_start=0,
):
    if len(scenes) < 2:
        raise Exception("Need at least two scenes")

    print("Generating tiles")

    for triplet_n in tqdm(range(N_triplets)[N_start:]):
        # sample different datasets
        tn_target, tn_dist = random.sample(range(len(scenes)), 2)
        da_target_scene = scenes[tn_target]
        da_distant_scene = scenes[tn_dist]

        prefixes = "anchor neighbor distant".split(" ")

        output_files_exist = [
            os.path.exists(output_dir / TRIPLET_FN_FORMAT.format(triplet_n, p))
            for p in prefixes
        ]

        if all(output_files_exist):
            continue

        tiles_and_imgs = tiler.triplet_generator(
            da_target_scene=da_target_scene,
            da_distant_scene=da_distant_scene,
            tile_size=tile_size,
            tile_N=tile_N,
            tiling_bbox=tiling_bbox,
            neigh_dist_scaling=neighbor_distant_frac,
        )

        tiles, imgs = zip(*tiles_and_imgs)

        for (img, prefix) in zip(imgs, prefixes):
            fn_out = TRIPLET_FN_FORMAT.format(triplet_n, prefix)
            img.save(output_dir / fn_out, "PNG")

        meta = dict(
            target=dict(
                source_files=da_target_scene.attrs["source_files"],
                anchor=tiles[0].serialize_props(),
                neighbor=tiles[1].serialize_props(),
            ),
            distant=dict(
                source_files=da_distant_scene.attrs["source_files"],
                loc=tiles[2].serialize_props(),
            ),
        )

        meta_fn = output_dir / "{:05d}_meta.yaml".format(triplet_n)
        with open(meta_fn, "w") as fh:
            yaml.dump(meta, fh, default_flow_style=False)


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
