import luigi

from ...data.sources.satellite.rectpred import MakeAllRectRGBDataArrays
from ..eurec4a.flow import FullDatasetOpticalFlowTrajectories
from ...pipeline import XArrayTarget
from ...data.sources.satellite.pipeline import parse_scene_id


class TrajectoryEmbeddingSampling(luigi.Task):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    trajectories_path = luigi.Parameter()

    def requires(self):
        return dict(
            trajectories=luigi.LocalTarget(self.trajectories_path),
            images=MakeAllRectRGBDataArrays(
                dataset_path=self.dataset_path
            )
        )

    def _find_nearest_scene_id(self, time, scene_ids):
        times = [parse_scene_id(scene_id) for scene_id in scene_ids]
        raise NotImplementedError

    def run(self):
        ds_traj = self.input()['trajectories'].open()
        input_images = self.input()['images']

        for t in ds_traj.time.values:
            ds_points = ds_traj.sel(time=t)
            if 'scene_id' in ds_points:
                scene_id = ds_points.scene_id.item()
            else:
                scene_id = self._find_nearest_scene_id()
            img = input_images[scene_id].open()

            ds_points_stacked = ds_points.stack(n=ds_points.coords.keys())

            for n in ds_points_stacked.n.values:
                ds_point = ds_points_stacked.sel(n=n)
                import ipdb
                ipdb.set_trace()
                a = 2

    def output(self):
        fn = "trajectory_embeddings.nc"
        return XArrayTarget(fn)


class FlowTrajectoryEmbeddingSampling(TrajectoryEmbeddingSampling):
    @property
    def trajectories_path(self):
        pass

    def requires(self):
        tasks = super().requires()
        tasks['trajectories'] = FullDatasetOpticalFlowTrajectories(
            dataset_path=self.dataset_path
        )
        return tasks
