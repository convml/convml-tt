import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.append('../../')
from convml_tt.data.sources import satdata

import importlib
importlib.reload(satdata.processing)
importlib.reload(satdata.tiler)
importlib.reload(satdata.utils)

import satpy
satpy.scene.LOG.propogate = False

TILING_BBOX_EXTENT = [
    (-60, 10), # (lon, lat) min
    (-30, 30) # (lon, lat) max
]


def main():
    data_path = Path('../../../data/storage')

    # use center of tiling bbox for deciding central time
    lon_zenith = 0.5*(TILING_BBOX_EXTENT[1][0]+TILING_BBOX_EXTENT[0][0])
    dt_max = datetime.timedelta(hours=7)
    # dt_max = datetime.timedelta(minutes=15)

    tile_size = 200e3
    tile_N = 256

    N_triplets = 10000
    channels = [1,2,3,9]
    N_days = 7

    use_old = False
    max_workers=1

    def make_name():
        # script_name = Path(__file__).name.replace('.py', '')
        # return "{}__N{}_s{}".format(script_name, tile_N, tile_size)
        return "Nx{}_s{}_N{}_set2".format(tile_N, tile_size, N_triplets)

    tile_path = data_path/"tiles"/"goes16"/make_name()
    tile_path.mkdir(exist_ok=True, parents=True)

    path_composites = data_path/"composites"

    print("Tiles will be saved to `{}`".format(tile_path))


    t_zenith = satdata.calc_nearest_zenith_time_at_loc(lon_zenith) 
    times = [t_zenith - datetime.timedelta(days=n) for n in range(1, N_days)]

    cli = satdata.Goes16AWS()

    # get tuples of channel "keys" (storage ids) for all the valid times queried
    datasets_keys = satdata.processing.find_datasets_keys(
        times=times, dt_max=dt_max, cli=cli, channels=channels
    )

    # download all the files using the cli (flatting list first...)
    print("Downloading channel files...")
    keys = [fn for fns in datasets_keys for fn in fns]
    fns = cli.download(keys, output_dir=data_path/"sources"/"goes16")

    # now we know where each key was stored, so we can map the dataset keys to
    # the filenames they were stored into
    kfmap = dict(zip(keys, fns))
    datasets_filenames = [
        [kfmap[k] for k in dataset_keys] 
        for dataset_keys in datasets_keys
    ]

    # load "scenes", either a dataarray containing all channels for a given
    # time or a satpy channel (which is primed to create a RGB composite)
    print("Reading in files")
    scenes = satdata.processing.load_data_for_rgb(
        datasets_filenames=datasets_filenames, cli=cli, use_old=use_old,
        bbox_extent=TILING_BBOX_EXTENT, path_composites=path_composites
    )

    satdata.processing.generate_tile_triplets(
        scenes=scenes, tiling_bbox=TILING_BBOX_EXTENT,
        tile_size=tile_size,
        tile_N=tile_N, 
        N_triplets=N_triplets,
        output_dir=tile_path,
        max_workers=max_workers
    )

if __name__ == "__main__":
    main()
