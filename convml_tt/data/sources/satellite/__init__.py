import datetime
import yaml
from pathlib import Path

import satdata
import luigi

from ...dataset import TripletDataset
from . import processing, pipeline
from .bbox import LatLonBox


class SatelliteTripletDataset(TripletDataset):
    def __init__(self, domain_bbox, tile_size, tile_N, channels=[1,2,3],
                 **kwargs):
        """
        tile_size: size of tile [m]
        tile_N: dimension of tile [1]
        """
        super().__init__(**kwargs)
        self.domain_bbox = domain_bbox
        self.tile_size = tile_size
        self.tile_N = tile_N
        self.channels = channels

class FixedTimeRangeSatelliteTripletDataset(SatelliteTripletDataset):
    def __init__(self, t_start, N_days, N_hours_from_zenith,
                 **kwargs):
        super().__init__(**kwargs)
        self.t_start = t_start
        self.N_days = N_days
        self.N_hours_from_zenith = N_hours_from_zenith

        lon_zenith = kwargs['domain_bbox'][1][0]

        self._dt_max = datetime.timedelta(hours=N_hours_from_zenith)
        t_zenith = satdata.calc_nearest_zenith_time_at_loc(lon_zenith, t_ref=t_start) 
        self._times = [t_zenith + datetime.timedelta(days=n) for n in range(1, N_days+1)]

    def _get_tiles_base_path(self):
        return self.data_path/self.name

    def _get_dataset_train_study_split(self, cli):
        path_tiles_meta = self._get_tiles_base_path()/"training_study_split.yaml"

        if path_tiles_meta.exists():
            print("Definition for test vs study split was found in `{}`, "
                  "assuming that all necessary data has already been downloaded"
                  "".format(path_tiles_meta))
            with open(path_tiles_meta) as fh:
                datasets_filenames_split = yaml.load(fh)
        else:
            # get tuples of channel "keys" (storage ids) for all the valid times queried
            datasets_keys = processing.find_datasets_keys(
                times=self._times, dt_max=self._dt_max, channels=self.channels,
                cli=cli,
            )

            if len(datasets_keys) == 0:
                raise Exception("Couldn't find any data matching the provided query")

            # download all the files using the cli (flatting list first...)
            print("Downloading channel files...")
            keys = [fn for fns in datasets_keys for fn in fns]
            fns = cli.download(keys)

            # now we know where each key was stored, so we can map the dataset keys to
            # the filenames they were stored into
            kfmap = dict(zip(keys, fns))
            datasets_filenames_all = [
                [kfmap[k] for k in dataset_keys]
                for dataset_keys in datasets_keys
            ]

            datasets_filenames_split = (
                processing.pick_one_time_per_date_for_study(datasets_filenames_all)
            )

            with open(path_tiles_meta, "w") as fh:
                yaml.dump(datasets_filenames_split, fh, default_flow_style=False)

        return datasets_filenames_split

    def get_dataset_scene(self, data_path, scene_num, offline_cli=True, for_training=True):

        datasets_filenames_split = self._get_dataset_train_study_split(cli=cli)
        if for_training:
            datasets_filenames = datasets_filenames_split['train']
        else:
            datasets_filenames = datasets_filenames_split['study']

        scenes = processing.load_data_for_rgb(
            datasets_filenames=[datasets_filenames[scene_num],], cli=cli,
            bbox_extent=self.domain_bbox, path_composites=path_composites
        )
        return scenes[0]

    def generate(self, data_path, offline_cli):
        luigi.build([
            pipeline.GOES16Fetch(
                dt_max=self._dt_max,
                channels=[1,2,3],
                times=self._times,
                data_path=self.data_path
            )
        ], local_scheduler=True)

    # def generate(self, data_path, offline_cli):
        # local_storage_dir = data_path/"sources"/"goes16"
        # path_composites = data_path/"composites"

        # cli = satdata.Goes16AWS(
            # offline=offline_cli,
            # local_storage_dir=local_storage_dir
        # )

        # datasets_filenames_split = self._get_dataset_train_study_split(cli=cli)
        # self._generate(
            # data_path=data_path,
            # offline_clie=offline_cli,
            # datasets_filenames_split=datasets_filenames_split
        # )

    def get_domain(self):
        return LatLonBox(self.domain_bbox)

    def _generate(self, data_path, offline_cli, datasets_filenames_split):
        local_storage_dir = data_path/"sources"/"goes16"
        path_composites = data_path/"composites"

        cli = satdata.Goes16AWS(
            offline=offline_cli,
            local_storage_dir=local_storage_dir
        )

        for identifier, datasets_filenames in datasets_filenames_split.items():
            print("Creating tiles for `{}`".format(identifier))

            # load "scenes", either a dataarray containing all channels for a given
            # time or a satpy channel (which is primed to create a RGB composite)
            print("Reading in files")
            scenes = processing.load_data_for_rgb(
                datasets_filenames=datasets_filenames, cli=cli,
                bbox_extent=self.domain_bbox, path_composites=path_composites
            )

            tile_path = tile_path_base/identifier
            tile_path.mkdir(exist_ok=True)

            processing.generate_tile_triplets(
                scenes=scenes, tiling_bbox=TILING_BBOX_EXTENT,
                tile_size=tile_size,
                tile_N=tile_N,
                N_triplets=N_triplets[identifier],
                output_dir=tile_path,
                max_workers=max_workers
            )
