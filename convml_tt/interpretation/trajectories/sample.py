from pathlib import Path

import luigi
import numpy as np
import xarray as xr
from fastai.basic_train import load_learner
from fastai.vision import open_image
from tqdm import tqdm

from ...data.sources.satellite.pipeline import parse_scene_id
from ...data.sources.satellite.rectpred import MakeAllRectRGBDataArrays
from ...pipeline import XArrayTarget
from ..eurec4a.data import N_TILE, FakeImagesList
from ..eurec4a.flow import FullDatasetOpticalFlowTrajectories


class RectTilerPoints:
    def __init__(self, img, N_tile):
        self.img = img
        self.ny, self.nx = img.size
        self.nxt, self.nyt = N_tile

    @staticmethod
    def _crop_fastai_im(img, i, j, nx=256, ny=256):
        img_copy = img.__class__(img._px[:, j : j + ny, i : i + nx])
        # From PIL docs: The crop rectangle, as a (left, upper, right, lower)-tuple.
        return img_copy

    def __call__(self, i_center, j_center):
        # half-widths
        nxt_hw = self.nxt // 2
        nyt_hw = self.nyt // 2
        if i_center < nxt_hw or i_center > (self.nx - nxt_hw):
            return None
        elif j_center < nyt_hw or j_center > (self.ny - nyt_hw):
            return None
        else:
            return self._crop_fastai_im(
                img=self.img,
                i=i_center - nxt_hw,
                j=j_center - nyt_hw,
                nx=self.nxt,
                ny=self.nyt,
            )


def make_tile_predictions(model, tile_images):
    il = FakeImagesList(None, None)

    for tile_img in tile_images:
        il.append(tile_img)

    return np.stack(model.predict(il)[1])


class TrajectoryEmbeddingSampling(luigi.Task):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    trajectories_path = luigi.Parameter()

    def requires(self):
        return dict(images=MakeAllRectRGBDataArrays(dataset_path=self.dataset_path))

    def _find_nearest_scene_id(self, time, scene_ids):
        # times = [parse_scene_id(scene_id) for scene_id in scene_ids]
        raise NotImplementedError

    def run(self):
        ds_traj = self.input()["trajectories"].open()
        input_images = self.input()["images"]

        model_fullpath = Path(self.model_path)
        model_path, model_fn = model_fullpath.parent, model_fullpath.name
        model = load_learner(model_path, model_fn)

        if "scene_id" in ds_traj.coords:
            scene_ids = ds_traj.scene_id.values
        else:
            raise NotImplementedError

        emb_dataarrays = []
        for scene_id in tqdm(scene_ids):
            ds_points = ds_traj.sel(scene_id=scene_id)
            image_path = input_images[scene_id].fn
            img = open_image(image_path)
            tile_sampler = RectTilerPoints(img=img, N_tile=N_TILE)

            ds_points_stacked = ds_points.stack(n=ds_points.dims.keys())

            N_points = int(ds_points_stacked.n.count())

            tile_images = []
            has_valid_tile = np.zeros(N_points).astype(bool)
            for n in ds_points_stacked.n.values:
                ds_point = ds_points_stacked.sel(n=n)
                tile_img = tile_sampler(
                    i_center=ds_point.i.values, j_center=ds_point.j.values
                )
                if tile_img:
                    has_valid_tile[n] = True
                    tile_images.append(tile_img)

            N_dims = 100
            embs_all = np.nan * np.ones((N_points, N_dims))

            if len(tile_images) > 0:
                embs = make_tile_predictions(model=model, tile_images=tile_images)
                N_embs, N_dims_ = embs.shape
                assert N_dims == N_dims_
                assert np.count_nonzero(has_valid_tile) == N_embs
                embs_all[has_valid_tile] = embs
            else:
                embs = None

            da_embs_scene = xr.DataArray(
                embs_all,
                coords=ds_points_stacked.coords,
                dims=list(ds_points_stacked.dims)
                + [
                    "emb_dim",
                ],
            ).unstack("n")

            emb_dataarrays.append(da_embs_scene)

        da_embs = xr.concat(emb_dataarrays, dim="scene_id")
        times = [parse_scene_id(scene_id) for scene_id in da_embs.scene_id.values]
        da_embs.coords["time"] = ("scene_id",), times
        da_embs.to_netcdf(self.output().fn)

    def output(self):
        fn = "trajectory_embeddings.nc"
        return XArrayTarget(fn)


class FlowTrajectoryEmbeddingSampling(TrajectoryEmbeddingSampling):
    @property
    def trajectories_path(self):
        pass

    def requires(self):
        tasks = super().requires()
        tasks["trajectories"] = FullDatasetOpticalFlowTrajectories(
            dataset_path=self.dataset_path
        )
        return tasks
