import subprocess
from collections import OrderedDict
from pathlib import Path

import luigi
import numpy as np
import xarray as xr
from convml_data import DataSource
from convml_data.pipeline import (
    GroupedSceneBulkProcessingBaseTask,
    SceneBulkProcessingBaseTask,
    SceneRegriddedData,
    parse_scene_id,
)

from ....pipeline import XArrayTarget
from ..flow.calc import extract_trajectories
from ..flow.plot import plot_scene_trajectories


class ComputeOpticalFlowTrajectories(luigi.Task):
    """
    Compute optical flow trajectories for a set of scene IDs that will be
    assumed to be consecutive in time
    """

    scene_ids = luigi.ListParameter()
    data_path = luigi.Parameter()
    prefix = luigi.Parameter()
    max_num_trajectories = luigi.IntParameter(default=400)
    min_point_distance = luigi.IntParameter(default=100)

    def requires(self):
        tasks = OrderedDict()
        for scene_id in self.scene_ids:  # noqa
            tasks[scene_id] = SceneRegriddedData(
                data_path=self.data_path,
                scene_id=scene_id,
            )

        return tasks

    def run(self):
        input = self.input()
        image_filenames = [t["image"].path for t in input.values()]
        ds_trajs = extract_trajectories(
            image_filenames=image_filenames,
            point_method_kwargs=dict(
                min_distance=self.min_point_distance,
                max_corners=self.max_num_trajectories,
            ),
        )

        # add a `scene_id` coordinate and make it the primary one
        fn_to_scene_id = dict(
            [(t["image"].fn, scene_id) for (scene_id, t) in input.items()]
        )
        scene_ids = [fn_to_scene_id[fn] for fn in ds_trajs.image_filename.values]
        ds_trajs["scene_id"] = "image_filename", scene_ids
        ds_trajs = ds_trajs.swap_dims(dict(image_filename="scene_id"))

        datasets_posns = []
        for scene_id in ds_trajs.scene_id.values:
            ds_points_scene = ds_trajs.sel(scene_id=scene_id)
            i_points = ds_points_scene.i.values
            j_points = ds_points_scene.j.values

            da_imgdata = self.input()[scene_id]["data"].open()
            lat_points = da_imgdata.lat.values[i_points, j_points]
            lon_points = da_imgdata.lon.values[i_points, j_points]
            x_points = da_imgdata.x.values[i_points]
            y_points = da_imgdata.y.values[j_points]

            m_nans = np.logical_or(i_points == -1, j_points == -1)

            def set_nans(v):
                return np.where(m_nans, np.nan, v)

            ds_points_scene["x"] = ("traj_id",), set_nans(x_points)
            ds_points_scene["y"] = ("traj_id",), set_nans(y_points)
            ds_points_scene["lat"] = ("traj_id",), set_nans(lat_points)
            ds_points_scene["lon"] = ("traj_id",), set_nans(lon_points)

            datasets_posns.append(ds_points_scene)

        ds_trajs = xr.concat(datasets_posns, dim="scene_id")
        for v in ["x", "y", "lat", "lon"]:
            ds_trajs[v].attrs.update(da_imgdata[v].attrs)
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        ds_trajs.to_netcdf(self.output().fn)

    def output(self):
        fn = f"{self.prefix}.flow_trajectories.nc"
        p_out = Path(self.data_path) / "rect" / "trajectories" / fn
        return XArrayTarget(str(p_out))


class DatasetPrefixOpticalFlowTrajectories(SceneBulkProcessingBaseTask):
    """
    Compute optical flow trajectories for all scenes sharing a common prefix
    """

    scene_prefix = luigi.Parameter()
    # this won't actually be used because we'll only need one single parent
    # task - the one that computes the trajectories for the scene collection
    TaskClass = ComputeOpticalFlowTrajectories

    @property
    def scene_filter(self):
        return f"{self.scene_prefix}.*"

    def _build_runtime_tasks(self):
        all_source_data = self.input().read()

        scene_ids = all_source_data.keys()
        scene_ids = self._filter_scene_ids(scene_ids=scene_ids)

        task = self.TaskClass(
            scene_ids=scene_ids,
            data_path=self.data_path,
            prefix=self.scene_prefix,
        )
        return task

    def output(self):
        if not self.input().exists():
            # the fetch has not completed yet, so we don't know how many scenes
            # we will be working on. Therefore we just return a target we know
            # will newer exist
            return luigi.LocalTarget("__fake_file__.nc")

        task = self._build_runtime_tasks()
        return task.output()


class GroupedDatasetOpticalFlowTrajectories(GroupedSceneBulkProcessingBaseTask):
    TaskClass = ComputeOpticalFlowTrajectories

    def _get_task_class_kwargs(self):
        return {}

    def run(self):
        yield super().run()

        tasks = self._build_runtime_tasks()
        datasets = [t.output().open() for t in tasks.values()]

        ds = xr.concat(datasets, dim="scene_id")
        times = [parse_scene_id(scene_id) for scene_id in ds.scene_id.values]
        ds.coords["time"] = ("scene_id",), times
        ds.to_netcdf(self.output().fn)

    def output(self):
        fn = "flow_trajectories_all.nc"
        return XArrayTarget(fn)


class PlotSceneWithTrajectories(luigi.Task):
    """
    Create trajectories using all scenes defined by `trajectory_scene_ids` and create a
    plot of the scene_image with the trajectories overlaid
    """

    trajectory_scene_ids = luigi.ListParameter()
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter()
    prefix = luigi.Parameter()
    max_num_trajectories = luigi.IntParameter(default=400)
    dt_max = luigi.IntParameter(default=120)  # [min]
    min_point_distance = luigi.IntParameter(default=100)

    def requires(self):
        reqs = {}
        reqs["trajectories"] = ComputeOpticalFlowTrajectories(
            scene_ids=self.trajectory_scene_ids,
            data_path=self.data_path,
            prefix=self.prefix,
            max_num_trajectories=self.max_num_trajectories,
            min_point_distance=self.min_point_distance,
        )
        reqs["scene"] = SceneRegriddedData(
            data_path=self.data_path, scene_id=self.scene_id
        )
        return reqs

    def run(self):
        ds_traj = self.input()["trajectories"].open()
        data_source = DataSource.load(self.data_path)
        ax = plot_scene_trajectories(
            ds_traj=ds_traj,
            scene_id=self.scene_id,
            data_source=data_source,
            dt_max=np.timedelta64(self.dt_max, "m"),
        )
        fig = ax.figure
        self.output().makedirs()
        fig.savefig(self.output().path)

    def output(self):
        fn = f"{self.prefix}.{self.scene_id}.flow_trajectories.png"
        p_out = (
            Path(self.data_path)
            / "rect"
            / "trajectories"
            / "animations"
            / "frames"
            / fn
        )
        return XArrayTarget(str(p_out))


class PlotScenesWithScenePrefixTrajectories(luigi.Task):
    """
    For the set of scenes that share a common prefix compute trajectories of
    movement using optical flow and plot each scene with the trajectories
    overlaid
    """

    prefix = luigi.Parameter()
    scene_ids = luigi.ListParameter()
    max_num_trajectories = luigi.IntParameter(default=400)
    min_point_distance = luigi.IntParameter(default=100)
    dt_max = luigi.IntParameter(default=120)  # [min]
    data_path = luigi.Parameter()
    create_animation = luigi.BoolParameter(default=False)

    def requires(self):
        tasks = []
        for scene_id in self.scene_ids:
            task = PlotSceneWithTrajectories(
                scene_id=scene_id,
                trajectory_scene_ids=self.scene_ids,
                max_num_trajectories=self.max_num_trajectories,
                min_point_distance=self.min_point_distance,
                prefix=self.prefix,
                dt_max=self.dt_max,
                data_path=self.data_path,
            )
            tasks.append(task)
        return tasks

    def run(self):
        if self.create_animation:
            args = ["convert"]
            args += [o.path for o in self.output()["frames"]]
            args.append(self.output()["animation"].path)
            proc = subprocess.Popen(args=args)
            stdout, stderr = proc.communicate()
            if stderr is not None:
                raise Exception(stderr)

    def output(self):
        parent_output = self.input()
        if not self.create_animation:
            return parent_output

        fn = f"{self.prefix}.gif"
        p_out = Path(self.data_path) / "rect" / "trajectories" / "animations" / fn
        return dict(frames=parent_output, animation=luigi.LocalTarget(p_out))


class PlotAllScenesWithScenePrefixTrajectories(GroupedSceneBulkProcessingBaseTask):
    TaskClass = PlotScenesWithScenePrefixTrajectories
    max_num_trajectories = luigi.IntParameter(default=400)
    min_point_distance = luigi.IntParameter(default=100)
    dt_max = luigi.IntParameter(default=120)  # [min]
    create_animations = luigi.BoolParameter(default=True)

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            max_num_trajectories=self.max_num_trajectories,
            dt_max=self.dt_max,
            min_point_distance=self.min_point_distance,
            create_animation=self.create_animations,
        )
