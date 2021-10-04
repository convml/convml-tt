import datetime

from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import satdata
import luigi
import xarray as xr
import numpy as np

from ...dataset import TripletDataset, TrajectoryDataset
from . import processing, pipeline
from .bbox import LatLonBox
from .tiler import RectTile


def _ensure_task_run(t):
    if not t.output().exists():
        luigi.build(
            [
                t,
            ],
            local_scheduler=True,
        )
    if not t.output().exists():
        raise Exception("Task didn't complete")


class SatelliteDatasetMixin:
    def fetch_source_data(self):
        return pipeline.GOES16Fetch(
            dt_max=self._dt_max,
            channels=self.channels,
            times=self._times,
            data_path=self.data_path,
        )


class SatelliteTripletDataset(TripletDataset, SatelliteDatasetMixin):
    def __init__(self, domain_bbox, tile_size, tile_N, channels=[1, 2, 3], **kwargs):
        """
        tile_size: size of tile [m]
        tile_N: dimension of tile [1]
        """
        super().__init__(**kwargs)
        self.tile_size = tile_size
        self.tile_N = tile_N
        self.channels = channels
        self.domain_bbox = domain_bbox


class SatelliteTrajectoryDataset(TrajectoryDataset, SatelliteDatasetMixin):
    # GOES-16 data is every 10min now, so 5min before/after should ensure we
    # get one and only one scene
    _dt_max = datetime.timedelta(minutes=5)
    channels = [1, 2, 3]

    def __init__(self, tile_size, tile_N, channels=[1, 2, 3], **kwargs):
        """
        tile_size: size of tile [m]
        tile_N: dimension of tile [1]
        """
        super().__init__(**kwargs)
        self.tile_size = tile_size
        self.tile_N = tile_N
        self.channels = channels

    def get_domain_rect(self, da_scene):
        ds_pt = self._ds_traj.sel(time=da_scene.start_time, method="nearest")
        lat0, lon0 = ds_pt.lat.item(), ds_pt.lon.item()
        tile = RectTile(
            lat0=lat0, lon0=lon0, l_meridional=self.tile_size, l_zonal=self.tile_size
        )
        return tile

    def get_domain(self, da_scene):
        tile = self.get_domain_rect(da_scene=da_scene)
        return LatLonBox(tile.get_bounds())


class FixedTimeRangeSatelliteTripletDataset(SatelliteTripletDataset):
    class SourceDataNotDownloaded(Exception):
        pass

    def __init__(self, t_start, N_days, N_hours_from_zenith, **kwargs):
        if "domain_bbox" not in kwargs and "domain" in kwargs:
            domain_rect = RectTile(**kwargs["domain"])
            domain_bounds = domain_rect.get_bounds()
            kwargs["domain_bbox"] = [
                [domain_bounds[:, 0].min(), domain_bounds[:, 1].min()],  # noqa
                [domain_bounds[:, 0].max(), domain_bounds[:, 1].max()],  # noqa
            ]
            kwargs["extra"]["rectpred"]["domain"] = kwargs.pop("domain")
        super().__init__(**kwargs)
        self.t_start = t_start
        self.N_days = N_days
        self.N_hours_from_zenith = N_hours_from_zenith

        lon_zenith = kwargs["domain_bbox"][1][0]

        self._dt_max = datetime.timedelta(hours=N_hours_from_zenith)
        t_zenith = satdata.calc_nearest_zenith_time_at_loc(lon_zenith, t_ref=t_start)
        self._times = [
            t_zenith + datetime.timedelta(days=n) for n in range(1, N_days + 1)
        ]

    def _get_tiles_base_path(self):
        return self.data_path / self.name

    def _get_dataset_train_study_split(self):
        return pipeline.StudyTrainSplit(
            dt_max=self._dt_max,
            channels=[1, 2, 3],
            times=self._times,
            data_path=self.data_path,
        )

    def get_num_scenes(self, for_training=True):
        t = self._get_dataset_train_study_split()
        _ensure_task_run(t)
        datasets_filenames_split = t.output().read()

        if for_training:
            datasets_filenames = datasets_filenames_split["train"]
        else:
            datasets_filenames = datasets_filenames_split["study"]

        return len(datasets_filenames)

    def get_scene(self, scene_num, for_training=True):
        t = self._get_dataset_train_study_split()
        if not t.output().exists():
            raise self.SourceDataNotDownloaded("Please run `fetch_data`")

        datasets_filenames_split = t.output().read()
        if for_training:
            datasets_filenames = datasets_filenames_split["train"]
        else:
            datasets_filenames = datasets_filenames_split["study"]

        return pipeline.CreateRGBScene(
            source_fns=datasets_filenames[scene_num],
            domain_bbox=self.domain_bbox,
            data_path=self.data_path,
        )

    def generate(self):
        t = self._get_dataset_train_study_split()
        luigi.build([t], local_worker=True)
        datasets_filenames_split = t.output().read()

        local_storage_dir = self.data_path / "source_data" / "goes16"
        path_composites = self.data_path / "composites"

        for identifier, datasets_filenames in datasets_filenames_split.items():
            print("Creating tiles for `{}`".format(identifier))

            datasets_fullpaths = [
                [str(local_storage_dir / fn) for fn in dataset]
                for dataset in datasets_filenames
            ]

            # load "scenes", either a dataarray containing all channels for a
            # given time or a satpy channel (which is primed to create a RGB
            # composite)
            print("Reading in files")
            scenes = processing.load_data_for_rgb(
                datasets_filenames=datasets_fullpaths,
                datasource_cli=satdata.Goes16AWS,
                bbox_extent=self.domain_bbox,
                path_composites=path_composites,
            )

            tile_path_base = self._get_tiles_base_path()
            tile_path = tile_path_base / identifier
            tile_path.mkdir(exist_ok=True)

            processing.generate_tile_triplets(
                scenes=scenes,
                tiling_bbox=self.domain_bbox,
                tile_size=self.tile_size,
                tile_N=self.tile_N,
                N_triplets=self.N_triplets[identifier],
                output_dir=tile_path,
            )

    def get_domain(self, **kwargs):
        return LatLonBox(self.domain_bbox)

    def get_domain_rect(self, **kwargs):
        return RectTile(**self.extra["rectpred"]["domain"])
