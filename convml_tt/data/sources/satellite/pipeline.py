from pathlib import Path

import satdata
import luigi
import yaml
import dateutil.parser
import xarray as xr
import numpy as np

from . import processing, satpy_rgb, tiler, bbox

SOURCE_DIR = Path("source_data")/"goes16"

class YAMLTarget(luigi.LocalTarget):
    def write(self, obj):
        with self.open('w') as fh:
            yaml.dump(obj, fh)

    def read(self):
        with self.open() as fh:
            return yaml.load(fh)

class DatetimeListParameter(luigi.Parameter):
    def parse(self, x):
        return [dateutil.parser.parse(s) for s in x.split(',')]

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

        filenames = cli.query(time=self.time, region='F', debug=self.debug,
                              channel=self.channel, dt_max=self.dt_max)

        Path(self.output().fn).parent.mkdir(exist_ok=True)
        self.output().write(filenames)

    def output(self):
        fn = 'source_data/ch{}_keys_{}.yaml'.format(
                          self.channel, self.time.isoformat())
        p = Path(self.data_path)/fn
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
                GOES16Query(dt_max=self.dt_max, channel=c, time=t)
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
            raise Exception("There are a different number of files for the"
                            " channels requested")

        local_storage_dir = Path(self.data_path).expanduser()/SOURCE_DIR
        cli = satdata.Goes16AWS(
            offline=self.offline_cli,
            local_storage_dir=local_storage_dir
        )

        # download all the files using the cli (flatting list first...)
        all_files = [fn for fns in files_per_channel.values() for fn in fns]
        print("Downloading channel files...")
        fns = cli.download(all_files)

        file_sets = [list(a) for a in zip(*files_per_channel.values())]
        Path(self.output().fn).parent.mkdir(exist_ok=True)
        self.output().write(file_sets)

    def output(self):
        fn = 'source_data/all_files.yaml'
        p = Path(self.data_path)/fn
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
        datasets_filenames_split = (
            processing.pick_one_time_per_date_for_study(
                datasets_filenames_all,
                datasource_cli=satdata.Goes16AWS
            )
        )
        Path(self.output().fn).parent.mkdir(exist_ok=True)
        self.output().write(datasets_filenames_split)

    def output(self):
        fn = "source_data/training_study_split.yaml"
        p = Path(self.data_path)/fn
        return YAMLTarget(str(p))

class RGBCompositeNetCDFFile(luigi.LocalTarget):
    def save(self, da_truecolor, source_fns):
        Path(self.fn).parent.mkdir(exist_ok=True, parents=True)
        satpy_rgb.save_scene_meta(source_fns=source_fns, fn_meta=self.path_meta)
        da_truecolor.to_netcdf(self.fn)

    @property
    def path_meta(self):
        return self.fn.replace('.nc', '.meta.yaml')

    def open(self):
        try:
            da = xr.open_dataarray(self.fn)
        except:
            print("Error opening `{}`".format(self.fn))
            raise
        meta_info = satpy_rgb.load_scene_meta(fn_meta=self.path_meta)
        da.attrs.update(meta_info)

        return da

class CreateRGBScene(luigi.Task):
    """
    Create RGB composite scene from GOES-16 radiance files
    """
    source_fns = luigi.ListParameter()
    domain_bbox = luigi.ListParameter()
    domain_bbox_pad_frac = luigi.FloatParameter(default=0.05)
    data_path = luigi.Parameter()

    def run(self):
        scene_fns = [
            str(Path(self.data_path)/SOURCE_DIR/fn) for fn in self.source_fns
        ]
        da_truecolor = satpy_rgb.load_rgb_files_and_get_composite_da(
            scene_fns=scene_fns
        )

        bbox_domain = bbox.LatLonBox(self.domain_bbox)

        da_truecolor_domain = tiler.crop_field_to_latlon_box(
            da=da_truecolor, box=np.array(bbox_domain.get_bounds()).T,
            pad_pct=self.domain_bbox_pad_frac
        )

        da_truecolor_domain = satpy_rgb._cleanup_composite_da_attrs(da_truecolor_domain)

        self.output().save(da_truecolor=da_truecolor_domain,
                           source_fns=scene_fns)

    def output(self):
        fn_da = satpy_rgb.make_composite_filename(
            scene_fns=self.source_fns, bbox_extent=self.domain_bbox
        )
        p = Path(self.data_path)/"composites"/fn_da
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
        datasets_filenames_split = self.input().read()

    def output(self):
        return YAMLTarget("training_study_split.yaml")