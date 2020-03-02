import datetime
import yaml
from pathlib import Path

import satdata
import luigi

from ...dataset import TripletDataset
from . import processing, pipeline
from .bbox import LatLonBox

def _ensure_task_run(t):
    if not t.output().exists():
        luigi.build([t, ], local_scheduler=True)
    if not t.output().exists():
        raise Exception("Task didn't complete")

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

    def _get_dataset_train_study_split(self):
        t = pipeline.StudyTrainSplit(
                dt_max=self._dt_max,
                channels=[1,2,3],
                times=self._times,
                data_path=self.data_path
            )
        _ensure_task_run(t)
        return t.output().read()

    def get_scene(self, scene_num, for_training=True):
        datasets_filenames_split = self._get_dataset_train_study_split()
        if for_training:
            datasets_filenames = datasets_filenames_split['train']
        else:
            datasets_filenames = datasets_filenames_split['study']

        t = pipeline.CreateRGBScene(
            source_fns=datasets_filenames[scene_num],
            domain_bbox=self.domain_bbox,
            data_path=self.data_path,
        )
        _ensure_task_run(t)
        return t.output().open()

    def fetch_source_data(self):
        t = pipeline.GOES16Fetch(
                dt_max=self._dt_max,
                channels=[1,2,3],
                times=self._times,
                data_path=self.data_path
            )
        _ensure_task_run(t)
        return t.output().read()

    def generate(self):
        datasets_filenames_split = self._get_dataset_train_study_split()

        local_storage_dir = source_data_path/"source_data"/"goes16"
        path_composites = source_data_path/"composites"

        for identifier, datasets_filenames in datasets_filenames_split.items():
            print("Creating tiles for `{}`".format(identifier))

            datasets_fullpaths = [
                [str(local_storage_dir/fn) for fn in dataset]
                for dataset in datasets_filenames
            ]

            # load "scenes", either a dataarray containing all channels for a given
            # time or a satpy channel (which is primed to create a RGB composite)
            print("Reading in files")
            scenes = processing.load_data_for_rgb(
                datasets_filenames=datasets_fullpaths,
                datasource_cli=satdata.Goes16AWS,
                bbox_extent=self.domain_bbox, path_composites=path_composites
            )

            tile_path_base = self._get_tiles_base_path()
            tile_path = tile_path_base/identifier
            tile_path.mkdir(exist_ok=True)

            processing.generate_tile_triplets(
                scenes=scenes, tiling_bbox=self.domain_bbox,
                tile_size=self.tile_size,
                tile_N=self.tile_N,
                N_triplets=self.N_triplets[identifier],
                output_dir=tile_path,
            )

    def get_domain(self):
        return LatLonBox(self.domain_bbox)
