from pathlib import Path

import satdata
import luigi
import dateutil.parser
import xarray as xr
import numpy as np
import isodate

from ....pipeline import YAMLTarget


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

    def get_time(filename):
        return satdata.Goes16AWS.parse_key(filename, parse_times=True)["start_time"]

    def run(self):
        cli = satdata.Goes16AWS(offline=False)

        filenames = cli.query(
            time=self.time,
            region="F",
            debug=self.debug,
            channel=self.channel,
            dt_max=self.dt_max,
        )

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(filenames)

    def output(self):
        fn = "ch{}_keys_{}_{}.yaml".format(
            self.channel, self.time.isoformat(), isodate.duration_isoformat(self.dt_max)
        )
        p = Path(self.data_path) / fn
        return YAMLTarget(str(p))


class GOES16Fetch(luigi.Task):
    keys = luigi.ListParameter()
    data_path = luigi.Parameter()
    offline_cli = luigi.BoolParameter(default=False)

    @property
    def cli(self):
        local_storage_dir = Path(self.data_path).expanduser()
        return satdata.Goes16AWS(
            offline=self.offline_cli, local_storage_dir=local_storage_dir
        )

    def run(self):
        self.cli.download(list(self.keys))

    def output(self):
        targets = [
            luigi.LocalTarget(str(Path(self.data_path) / key)) for key in self.keys
        ]
        return targets


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


class GenerateTripletsOld(luigi.Task):
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
