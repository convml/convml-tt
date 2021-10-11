import luigi
from pathlib import Path
import datetime
import logging
import itertools

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
                f"Only {len(timestamp_files)} were found for timestamp {timestamp}"
                " so this timestamp will be excluded"
            )

    return scene_filesets


def get_time_for_filename(data_source, filename):
    if data_source.source == "goes16":
        return GOES16Query.get_time(filename=filename)
    elif data_source.source == "LES":
        return FindLESFiles.get_time(filename=filename)
    else:
        raise NotImplementedError(data_source)


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
        data_source = self.data_source
        source_data_path = Path(self.data_path) / "source_data" / data_source.source

        tasks = None
        if data_source.source == "goes16":
            if data_source.type == "truecolor_rgb":
                tasks = {}
                for t_start, t_end in data_source.time_intervals:
                    dt_total = t_end - t_start
                    t_center = t_start + dt_total / 2.0

                    for channel in [1, 2, 3]:
                        t = GOES16Query(
                            data_path=source_data_path,
                            time=t_center,
                            dt_max=dt_total / 2.0,
                            channel=channel,
                        )
                        tasks.setdefault(channel, []).append(t)
            else:
                raise NotImplementedError(data_source.type)
        elif data_source.source == "LES":
            kind, *variables = data_source.type.split("__")
            if not kind == "singlechannel":
                raise NotImplementedError(data_source.type)
            else:
                source_variable = variables[0]

            filename_glob = (
                data_source.files is not None and data_source.files or "*.nc"
            )
            tasks = FindLESFiles(
                data_path=source_data_path,
                source_variable=source_variable,
                filename_glob=filename_glob,
            )
        else:
            raise NotImplementedError(data_source.source)

        return tasks

    def run(self):
        data_source = self.data_source
        scenes = {}

        input = self.input()
        if type(input) == dict:
            channels_and_filenames = OrderedDict()
            if data_source.type == "truecolor_rgb":
                channel_order = [1, 2, 3]
            else:
                raise NotImplementedError(data_source.type)

            opened_inputs = {}
            for input_name, input_parts in input.items():
                opened_inputs[input_name] = list(
                    itertools.chain(*[input_part.open() for input_part in input_parts])
                )

            for channel in channel_order:
                channels_and_filenames[channel] = opened_inputs[channel]

            scene_sets = merge_multichannel_sources(
                channels_and_filenames, time_fn=self.get_time_for_filename
            )

            for scene_filenames in scene_sets:
                t_scene = get_time_for_filename(filename=scene_filenames[0], data_source=data_source)
                if data_source.filter_scene_times(t_scene):
                    scene_id = make_scene_id(source=data_source.source, t_scene=t_scene)
                    scenes[scene_id] = scene_filenames
        elif isinstance(input, YAMLTarget):
            scene_filenames = input.open()
            for scene_filename in scene_filenames:
                t_scene = get_time_for_filename(filename=scene_filename, data_source=data_source)
                if data_source.filter_scene_times(t_scene):
                    scene_id = make_scene_id(source=data_source.source, t_scene=t_scene)
                    scenes[scene_id] = scene_filename
        else:
            raise NotImplementedError(input)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(scenes)

    def output(self):
        ds = self.data_source
        fn = "scene_ids.yml"
        p = Path(self.data_path) / "source_data" / ds.source / ds.type / fn
        return YAMLTarget(str(p))


class DownloadAllSourceFiles(luigi.Task):
    """
    Download all source files for all scenes in the dataset
    """

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return GenerateSceneIDs(data_path=self.data_path)
