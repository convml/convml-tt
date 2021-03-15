from pathlib import Path

import luigi
import satdata
from .....utils.pipeline import YAMLTarget, DatetimeListParameter
from ...utils import SOURCE_DIR, create_scene_id


class GOES16Query(luigi.Task):
    """
    Use the GOES-16 CLI to query what files are available at a given time for a
    given channel
    """
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


def _check_num_files_per_channel(files_per_channel):
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

        def time_diff(idx0, idx1, fpc):
            # fpc: new sliced files_per_channel dictionary where each list
            # now has the same length
            channels = list(files_per_channel.keys())
            channel1 = channels[idx0]
            channel2 = channels[idx1]
            dt = get_time(fpc[channel1][0]) - get_time(fpc[channel2][0])
            return abs(dt.total_seconds())

        def timediff_all(fpc):
            return sum(
                [
                    time_diff(idx0, idx1, fpc)
                    for (idx0, idx1) in [(0, 1), (0, 2), (1, 2)]
                ]
            )

        n_max = max(num_files_per_channel.values())

        fpc1 = {
            c: len(fns) == n_max and fns[1:] or fns
            for (c, fns) in files_per_channel.items()
        }

        fpc2 = {
            c: len(fns) == n_max and fns[:-1] or fns
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


class GOES16Fetch(luigi.Task):
    """
    Fetch sets of channels (currently channels 1, 2 and 3) at max `dt_max` from
    each time in `times`
    """
    CLIClass = satdata.Goes16AWS

    dt_max = luigi.FloatParameter()
    # for now we always fetch the first three channels because we're making RGB composites
    channels = [1, 2, 3]
    times = DatetimeListParameter()
    data_path = luigi.Parameter()
    offline_cli = luigi.BoolParameter(default=False)

    def requires(self):
        if len(self.times) == 0:
            raise Exception("`times` argument must have non-zero length")

        reqs = {}
        for channel in self.channels:
            reqs[channel] = [
                GOES16Query(
                    dt_max=self.dt_max,
                    channel=channel,
                    time=t,
                    data_path=self.data_path,
                )
                for t in self.times
            ]
        return reqs

    def run(self):
        files_per_channel = {}
        for channel, queries in self.input().items():
            files_per_channel[channel] = []
            for query in queries:
                files_per_channel[channel] += query.read()

        _check_num_files_per_channel(files_per_channel=files_per_channel)

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
        return create_scene_id(datasource_name="goes16", t_scene=t)

    def output(self):
        fn = "source_data/all_files.yaml"
        p = Path(self.data_path) / fn
        return YAMLTarget(str(p))
