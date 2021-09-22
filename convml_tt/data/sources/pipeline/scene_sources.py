import luigi
from pathlib import Path
import datetime
import logging

from ..goes16.pipeline import GOES16Query
from ..les import FindLESFiles
from .. import DataSource
from ....pipeline import YAMLTarget
from collections import OrderedDict

log = logging.getLogger()


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
    channel_files_by_timestamp = {}
    N_channels = len(files_per_channel)
    for channel, channel_files in files_per_channel.items():
        for ch_filename in channel_files:
            file_timestamp = time_fn(ch_filename)
            time_group = channel_files_by_timestamp.setdefault(file_timestamp, {})
            time_group[channel] = ch_filename

    scene_filesets = []

    for timestamp in sorted(channel_files_by_timestamp.keys()):
        timestamp_files = channel_files_by_timestamp[timestamp]

        if len(timestamp_files) == N_channels:
            scene_filesets.append(
                [timestamp_files[channel] for channel in files_per_channel.keys()]
            )
        else:
            log.warn(
                "Only {len(timestamp_files)} were found for timestamp {timestamp}"
                " so this timestamp will be excluded"
            )

    return scene_filesets


class GenerateSceneIDs(luigi.Task):
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
        elif ds.source == "LES":
            kind, *variables = ds.type.split("__")
            if not kind == "singlechannel":
                raise NotImplementedError(ds.type)
            else:
                source_variable = variables[0]

            filename_glob = ds.files is not None and ds.files or "*.nc"
            tasks = FindLESFiles(
                data_path=source_data_path,
                source_variable=source_variable,
                filename_glob=filename_glob,
            )
        else:
            raise NotImplementedError(ds.source)

        return tasks

    def get_time_for_filename(self, filename):
        ds = self.data_source
        if ds.source == "goes16":
            return GOES16Query.get_time(filename=filename)
        elif ds.source == "LES":
            return FindLESFiles.get_time(filename=filename)
        else:
            raise NotImplementedError(ds.source)

    def run(self):
        ds = self.data_source
        scenes = {}

        input = self.input()
        if type(input) == dict:
            channels_and_filenames = OrderedDict()
            if ds.type == "truecolor_rgb":
                channel_order = [1, 2, 3]
            else:
                raise NotImplementedError(ds.type)

            opened_inputs = {
                input_name: input_item.open()
                for (input_name, input_item) in input.items()
            }

            for channel in channel_order:
                channels_and_filenames[channel] = opened_inputs[channel]

            scene_sets = merge_multichannel_sources(
                channels_and_filenames, time_fn=self.get_time_for_filename
            )

            for scene_filenames in scene_sets:
                t_scene = self.get_time_for_filename(filename=scene_filenames[0])
                scene_id = make_scene_id(source=ds.source, t_scene=t_scene)
                scenes[scene_id] = scene_filenames
        elif isinstance(input, YAMLTarget):
            scene_filenames = input.open()
            for scene_filename in scene_filenames:
                t_scene = self.get_time_for_filename(filename=scene_filename)
                scene_id = make_scene_id(source=ds.source, t_scene=t_scene)
                scenes[scene_id] = scene_filename
        else:
            raise NotImplementedError(input)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(scenes)

    def output(self):
        ds = self.data_source
        p = Path("source_data") / ds.source / ds.type
        fn = "scene_ids.yml"
        return YAMLTarget(str(p / fn))


class DownloadAllSourceFiles(luigi.Task):
    """
    Download all source files for all scenes in the dataset
    """

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return AllSceneIDs(data_path=self.data_path)
