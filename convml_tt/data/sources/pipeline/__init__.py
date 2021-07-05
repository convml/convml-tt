import luigi
from pathlib import Path
import itertools
import datetime

from ..satellite.pipeline import GOES16Query
from .. import DataSource
from ....pipeline import YAMLTarget

SCENE_ID_DATE_FORMAT = "%Y%m%d%H%M"


def load_meta(path):
    pass


def make_scene_id(source, t_scene):
    t_str = t_scene.strftime(SCENE_ID_DATE_FORMAT)
    return f"{source}__{t_str}"


def parse_scene_id(s):
    source, t_str = s.split("__")
    return source, datetime.datetime.strptime(t_str, SCENE_ID_DATE_FORMAT)


def merge_multichannel_sources(files_per_channel, time_fn):
    num_files_per_channel = {}
    for channel, files in files_per_channel.items():
        num_files_per_channel[channel] = len(files)

    if len(set(num_files_per_channel.values())) != 1:
        # the source data queries have resulted in a different number of
        # files being returned for the channels selected, probably because
        # the channels are not recorded at the same time and so one fell
        # outside the window

        def time_diff(filename1, filename2):
            """
            Look at the first set of filenames for the channels `c1` and `c2`
            and calculate the time difference
            """
            dt = time_fn(filename1) - time_fn(filename2)
            return abs(dt.total_seconds())

        def timediff_all(fpc):
            """
            For the files in the collection `fpc` calculate the time difference
            between all combinations of the first filename from each set
            """
            channels = list(files_per_channel.keys())
            return sum(
                [
                    time_diff(fpc[c1][0], fpc[c2][0])
                    for (c1, c2) in itertools.combinations(channels, 2)
                ]
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

    scenes_files = [list(a) for a in zip(*files_per_channel.values())]

    return scenes_files


class AllSceneIDs(luigi.Task):
    """
    Construct a "database" (actually a yaml-file) of all scene IDs in a dataset
    given the source, type and time-span of the dataset. Database contains a list
    of the scene IDs and the sourcefile(s) per scene.
    """

    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        ds = self.data_source
        source_data_path = Path(self.data_path) / "source_data" / ds.source

        tasks = None
        if ds.source == "goes16":
            if ds.type == "truecolor_rgb":
                tasks = {}
                t_start = ds.t_start
                t_end = ds.t_end
                dt_total = t_end - t_start
                t_center = t_start + dt_total / 2.0

                for channel in [1, 2, 3]:
                    t = GOES16Query(
                        data_path=source_data_path,
                        time=t_center,
                        dt_max=dt_total / 2.0,
                        channel=channel,
                    )
                    tasks[channel] = t
            else:
                raise NotImplementedError(ds.type)
        else:
            raise NotImplementedError(ds.source)

        return tasks

    def get_time_for_filename(self, filename):
        ds = self.data_source
        if ds.source == "goes16":
            return GOES16Query.get_time(filename=filename)
        else:
            raise NotImplementedError(ds.source)

    def run(self):
        ds = self.data_source
        scenes = {}

        inputs = self.input()
        if type(inputs) == dict:
            channels_and_filenames = {
                channel: inp.open() for (channel, inp) in inputs.items()
            }
            channels_and_filenames[1] = channels_and_filenames[1][1:]
            scene_sets = merge_multichannel_sources(
                channels_and_filenames, time_fn=self.get_time_for_filename
            )

            for scene_filenames in scene_sets:
                t_scene = self.get_time_for_filename(filename=scene_filenames[0])
                scene_id = make_scene_id(source=ds.source, t_scene=t_scene)
                scenes[scene_id] = scene_filenames
        else:
            raise NotImplementedError(inputs)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(scenes)

    def output(self):
        ds = self.data_source
        p = Path("source_data") / ds.source / ds.type
        fn = "scene_ids.yml"
        return YAMLTarget(str(p / fn))


class GenerateAllScenes(luigi.Task):
    data_path = luigi.Parameter(default=".")
