from pathlib import Path

import satdata
import luigi
import yaml
import dateutil.parser


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

    def run(self):
        cli = satdata.Goes16AWS(offline=False)

        filenames = cli.query(time=self.time, region='F', debug=self.debug,
                              channel=self.channel, dt_max=self.dt_max)

        self.output().write(filenames)

    def output(self):
        return YAMLTarget('source_data_keys_ch{}_{}.yaml'.format(
                          self.channel, self.time.isoformat()))

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

        local_storage_dir = Path(self.data_path).expanduser()/"sources"/"goes16"
        path_composites = Path(self.data_path).expanduser()/"composites"
        cli = satdata.Goes16AWS(
            offline=self.offline_cli,
            local_storage_dir=local_storage_dir
        )

        # download all the files using the cli (flatting list first...)
        all_files = [fn for fns in files_per_channel.values() for fn in fns]
        print("Downloading channel files...")
        fns = cli.download(all_files)

        file_sets = [list(a) for a in zip(*files_per_channel.values())]
        self.output().write(file_sets)

    def output(self):
        return YAMLTarget('source_data.yaml')

class StudyTrainSplit(luigi.Task):
    dt_max = luigi.FloatParameter()
    channels = luigi.ListParameter()
    times = luigi.ListParameter()

    def requires(self):
        return GOES16Fetch(
            dt_max=self.dt_max,
            channels=self.channels,
            times=self.times
        )

    def run(self):
        pass
