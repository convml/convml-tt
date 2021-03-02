from pathlib import Path

import satdata
import luigi
import dateutil.parser
import xarray as xr
import numpy as np
import datetime

from . import processing, satpy_rgb, tiler, bbox
from ....pipeline import YAMLTarget
from ...dataset import GenericDataset

SOURCE_DIR = Path("source_data")

SCENE_ID_DATE_FORMAT = "%Y%m%d%H%M"


def parse_scene_id(s):
    return datetime.datetime.strptime(s.replace("goes16_", ""), SCENE_ID_DATE_FORMAT)


class DatetimeListParameter(luigi.Parameter):
    def parse(self, x):
        return [dateutil.parser.parse(s) for s in x.split(",")]

    def serialize(self, x):
        return ",".join([t.isoformat() for t in x])


class GOES16Query(luigi.Task):
    dt_max = luigi.FloatParameter()
    channel = luigi.ListParameter()
    time = luigi.DateMinuteParameter()
    debug = luigi.BoolParameter(default=False)
    data_path = luigi.Parameter()

    def run(self):
        cli = satdata.Goes16AWS(offline=False)

        filenames = cli.query(
            time=self.time,
            region="F",
            debug=self.debug,
            channel=self.channel,
            dt_max=self.dt_max,
        )

        Path(self.output().fn).parent.mkdir(exist_ok=True)
        self.output().write(filenames)

    def output(self):
        fn = "source_data/ch{}_keys_{}.yaml".format(self.channel, self.time.isoformat())
        p = Path(self.data_path) / fn
        return YAMLTarget(str(p))


class GOES16Fetch(luigi.Task):
    dt_max = luigi.FloatParameter()
    channels = luigi.ListParameter()
    times = DatetimeListParameter()
    data_path = luigi.Parameter()
    offline_cli = luigi.BoolParameter(default=False)

    def requires(self):
        if len(self.times) == 0:
            raise Exception("`times` argument must have non-zero length")

        reqs = {}
        for c in self.channels:
            reqs[c] = [
                GOES16Query(
                    dt_max=self.dt_max, channel=c, time=t, data_path=self.data_path
                )
                for t in self.times
            ]
        return reqs

    def run(self):
        files_per_channel = {}
        for channel, queries in self.input().items():
            files_per_channel[channel] = []
            for qr in queries:
                files_per_channel[channel] += qr.read()

        num_files_per_channel = {}
        for channel, files in files_per_channel.items():
            num_files_per_channel[channel] = len(files)

        if len(set(num_files_per_channel.values())) != 1:
            # the source data queries have resulted in a different number of
            # files being returned for the channels selected, probably because
            # the channels are not recorded at the same time and so one fell
            # outside the window

            def get_time(fn):
                attrs = satdata.Goes16AWS.parse_key(fn, parse_times=True)
                return attrs["start_time"]

            def time_diff(i0, i1, fpc):
                # fpc: new sliced files_per_channel dictionary where each list
                # now has the same length
                channels = list(files_per_channel.keys())
                c1 = channels[i0]
                c2 = channels[i1]
                dt = get_time(fpc[c1][0]) - get_time(fpc[c2][0])
                return abs(dt.total_seconds())

            def timediff_all(fpc):
                return sum(
                    [time_diff(i0, i1, fpc) for (i0, i1) in [(0, 1), (0, 2), (1, 2)]]
                )

            N_max = max(num_files_per_channel.values())

            fpc1 = {
                c: len(fns) == N_max and fns[1:] or fns
                for (c, fns) in files_per_channel.items()
            }

            fpc2 = {
                c: len(fns) == N_max and fns[:-1] or fns
                for (c, fns) in files_per_channel.items()
            }

            if timediff_all(fpc1) < timediff_all(fpc2):
                files_per_channel = fpc1
            else:
                files_per_channel = fpc2

            # now double-check that we've got the right number of files
            num_files_per_channel = {}
            for channel, files in files_per_channel.items():
                num_files_per_channel[channel] = len(files)
            assert len(set(num_files_per_channel.values())) == 1

        local_storage_dir = Path(self.data_path).expanduser() / SOURCE_DIR
        cli = satdata.Goes16AWS(
            offline=self.offline_cli, local_storage_dir=local_storage_dir
        )

        # download all the files using the cli (flatting list first...)
        all_files = [fn for fns in files_per_channel.values() for fn in fns]
        print("Downloading channel files...")
        cli.download(all_files)

        scenes_files = [list(a) for a in zip(*files_per_channel.values())]

        indexed_scenes_files = {
            self._make_scene_id(scene_files): scene_files
            for scene_files in scenes_files
        }

        Path(self.output().fn).parent.mkdir(exist_ok=True)
        self.output().write(indexed_scenes_files)

    def _make_scene_id(self, files):
        attrs = satdata.Goes16AWS.parse_key(files[0], parse_times=True)
        t = attrs["start_time"]
        return "goes16_{}".format(t.strftime(SCENE_ID_DATE_FORMAT))

    def output(self):
        fn = "source_data/all_files.yaml"
        p = Path(self.data_path) / fn
        return YAMLTarget(str(p))


class StudyTrainSplit(luigi.Task):
    dt_max = luigi.FloatParameter()
    channels = luigi.ListParameter()
    times = DatetimeListParameter()
    data_path = luigi.Parameter()

    def requires(self):
        return GOES16Fetch(
            dt_max=self.dt_max,
            channels=self.channels,
            times=self.times,
            data_path=self.data_path,
        )

    def run(self):
        datasets_filenames_all = self.input().read()
        datasets_filenames_split = processing.pick_one_time_per_date_for_study(
            datasets_filenames_all, datasource_cli=satdata.Goes16AWS
        )
        Path(self.output().fn).parent.mkdir(exist_ok=True)
        self.output().write(datasets_filenames_split)

    def output(self):
        fn = "source_data/training_study_split.yaml"
        p = Path(self.data_path) / fn
        return YAMLTarget(str(p))


class RGBCompositeNetCDFFile(luigi.LocalTarget):
    def save(self, da_truecolor, source_fns):
        Path(self.fn).parent.mkdir(exist_ok=True, parents=True)
        satpy_rgb.save_scene_meta(source_fns=source_fns, fn_meta=self.path_meta)
        da_truecolor.to_netcdf(self.fn)

    @property
    def path_meta(self):
        return self.fn.replace(".nc", ".meta.yaml")

    def open(self):
        try:
            da = xr.open_dataarray(self.fn)
        except Exception:
            print("Error opening `{}`".format(self.fn))
            raise
        meta_info = satpy_rgb.load_scene_meta(fn_meta=self.path_meta)
        da.attrs.update(meta_info)

        return da


class CreateRGBScene(luigi.Task):
    """
    Create RGB composite scene from GOES-16 radiance files
    """

    scene_id = luigi.Parameter()
    dataset_path = luigi.Parameter()

    def requires(self):
        d = GenericDataset.load(self.dataset_path)
        return d.fetch_source_data()

    def run(self):
        d = GenericDataset.load(self.dataset_path)

        all_source_data = self.input().read()
        if self.scene_id not in all_source_data:
            raise Exception(
                "scene `{}` is missing from the source data file"
                "".format(self.scene_id)
            )
        else:
            scene_fns = [
                Path(self.dataset_path) / SOURCE_DIR / p
                for p in all_source_data[self.scene_id]
            ]
        # OBS: should probably check that the channels necessary for RGB scene
        # generation are the ones loaded here

        da_truecolor = satpy_rgb.load_rgb_files_and_get_composite_da(
            scene_fns=scene_fns
        )

        bbox_domain = d.get_domain(da_scene=da_truecolor)
        domain_bbox_pad_frac = getattr(d, "domain_bbox_pad_frac", 0.1)

        da_truecolor_domain = tiler.crop_field_to_latlon_box(
            da=da_truecolor,
            box=np.array(bbox_domain.get_bounds()).T,
            pad_pct=domain_bbox_pad_frac,
        )

        da_truecolor_domain = satpy_rgb._cleanup_composite_da_attrs(da_truecolor_domain)

        self.output().save(da_truecolor=da_truecolor_domain, source_fns=scene_fns)

    def output(self):
        fn = "{}.nc".format(self.scene_id)
        p = Path(self.dataset_path) / "composites" / "original_cropped" / fn
        t = RGBCompositeNetCDFFile(str(p))
        return t


class GenerateTriplets(luigi.Task):
    dt_max = luigi.FloatParameter()
    channels = luigi.ListParameter()
    times = DatetimeListParameter()
    data_path = luigi.Parameter()

    def requires(self):
        return StudyTrainSplit(
            dt_max=self.dt_max,
            channels=self.channels,
            times=self.times,
            data_path=self.data_path,
        )

    def run(self):
        pass

    def output(self):
        return YAMLTarget("training_study_split.yaml")
