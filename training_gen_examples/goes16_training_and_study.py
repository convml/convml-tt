"""
This script creates two dataset of triplets by splitting the input files into two parts
"""
import datetime
from pathlib import Path

import numpy as np
import yaml

import sys

sys.path.append("../../")
from convml_tt.data.sources import satdata

# import importlib
# importlib.reload(satdata.processing)
# importlib.reload(satdata.tiler)
# importlib.reload(satdata.utils)

import warnings

warnings.simplefilter("ignore")

import satpy

satpy.scene.LOG.propogate = False

TILING_BBOX_EXTENT = [(-60, 10), (-30, 30)]  # (lon, lat) min  # (lon, lat) max


def split_input_files(datasets_filenames):

    datasets_dates = []


def main():
    data_path = Path("../../../data/storage")

    # use center of tiling bbox for deciding central time
    lon_zenith = -60  # 0.5*(TILING_BBOX_EXTENT[1][0]+TILING_BBOX_EXTENT[0][0])
    dt_max = datetime.timedelta(hours=6)
    # dt_max = datetime.timedelta(minutes=15)

    tile_size = 200e3
    tile_N = 256

    N_triplets = dict(study=1000, train=10000)
    channels = [1, 2, 3, 9, 13]
    t_start = datetime.datetime(year=2018, month=12, day=1)
    N_days = 31 + 31 + 28  # until end of February

    use_old_composite_method = False
    max_workers = 1

    offline_cli = True

    path_composites = data_path / "composites"

    def make_name():
        return "Nx{}_s{}_N{}study_N{}train".format(
            tile_N, tile_size, N_triplets["study"], N_triplets["train"]
        )

    tile_path_base = data_path / "tiles" / "goes16" / make_name()
    tile_path_base.mkdir(exist_ok=True, parents=True)

    print("Tiles will be saved to `{}`".format(tile_path_base))

    path_tiles_meta = tile_path_base / "training_study_split.yaml"

    cli = satdata.Goes16AWS(
        local_storage_dir=data_path / "sources" / "goes16", offline=offline_cli
    )

    if path_tiles_meta.exists():
        print(
            "Definition for test vs study split was found in `{}`, "
            "assuming that all necessary data has already been downloaded"
            "".format(path_tiles_meta)
        )
        with open(path_tiles_meta) as fh:
            datasets_filenames_split = yaml.load(fh)
    else:
        t_zenith = satdata.calc_nearest_zenith_time_at_loc(lon_zenith, t_ref=t_start)
        times = [t_zenith + datetime.timedelta(days=n) for n in range(1, N_days)]

        # get tuples of channel "keys" (storage ids) for all the valid times queried
        datasets_keys = satdata.processing.find_datasets_keys(
            times=times, dt_max=dt_max, cli=cli, channels=channels
        )

        # download all the files using the cli (flatting list first...)
        print("Downloading channel files...")
        keys = [fn for fns in datasets_keys for fn in fns]
        fns = cli.download(keys)

        # now we know where each key was stored, so we can map the dataset keys to
        # the filenames they were stored into
        kfmap = dict(zip(keys, fns))
        datasets_filenames_all = [
            [kfmap[k] for k in dataset_keys] for dataset_keys in datasets_keys
        ]

        datasets_filenames_split = satdata.processing.pick_one_time_per_date_for_study(
            datasets_filenames_all
        )

        with open(path_tiles_meta, "w") as fh:
            yaml.dump(datasets_filenames_split, fh, default_flow_style=False)

    for identifier, datasets_filenames in datasets_filenames_split.items():
        print("Creating tiles for `{}`".format(identifier))

        # load "scenes", either a dataarray containing all channels for a given
        # time or a satpy channel (which is primed to create a RGB composite)
        print("Reading in files")
        scenes = satdata.processing.load_data_for_rgb(
            datasets_filenames=datasets_filenames,
            cli=cli,
            use_old=use_old_composite_method,
            bbox_extent=TILING_BBOX_EXTENT,
            path_composites=path_composites,
        )

        tile_path = tile_path_base / identifier
        tile_path.mkdir(exist_ok=True)

        satdata.processing.generate_tile_triplets(
            scenes=scenes,
            tiling_bbox=TILING_BBOX_EXTENT,
            tile_size=tile_size,
            tile_N=tile_N,
            N_triplets=N_triplets[identifier],
            output_dir=tile_path,
            max_workers=max_workers,
        )


if __name__ == "__main__":
    main()
