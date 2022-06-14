"""
luigi Tasks for producing embeddings with a trained neural network across a
whole dataset
"""
from pathlib import Path

import luigi
import numpy as np
import xarray as xr
from convml_data import DataSource
from convml_data.pipeline import SceneBulkProcessingBaseTask, SceneRegriddedData
from PIL import Image

from ....data.dataset import MovingWindowImageTilingDataset
from ....data.transforms import get_transforms as get_model_transforms
from ....pipeline import XArrayTarget
from ....system import TripletTrainerModel
from ....utils import get_embeddings

IMAGE_TILE_FILENAME_FORMAT = "{i0:05d}_{j0:05d}.png"


class ImagePredictionMapData(luigi.Task):
    """
    Make embeddings for a Cartesian 2D image using a specific model skipping
    every `step_size` grid-points in x and y. If `src_data_path` is provided
    then the attributes from that will be copied over (for example lat lon
    coordinates)
    """

    model_path = luigi.Parameter()
    image_path = luigi.Parameter()
    src_data_path = luigi.OptionalParameter()
    step_size = luigi.IntParameter(default=10)
    prediction_batch_size = luigi.IntParameter(default=32)

    def run(self):
        model = TripletTrainerModel.load_from_checkpoint(self.model_path)
        model.freeze()

        img = Image.open(self.image_path)

        N_tile = (256, 256)
        model_transforms = get_model_transforms(
            step="predict", normalize_for_arch=model.base_arch
        )
        ny_img, nx_img, _ = np.array(img).shape
        if nx_img < N_tile[0] or ny_img < N_tile[1]:
            raise Exception(
                "The requested scene image has too few pixels to contain "
                f"even a single tile (img size: {nx_img, ny_img} "
                f"vs tile size: {N_tile[0], N_tile[1]}), maybe the resolution "
                "is too coarse?"
            )

        tile_dataset = MovingWindowImageTilingDataset(
            img=img,
            transform=model_transforms,
            step=(self.step_size, self.step_size),
            N_tile=N_tile,
        )
        if len(tile_dataset) == 0:
            raise Exception("The provided tile-dataset doesn't contain any tiles! ")

        da_pred = get_embeddings(
            model=model,
            tile_dataset=tile_dataset,
            prediction_batch_size=self.prediction_batch_size,
        )

        if self.src_data_path:
            da_src = xr.open_dataarray(self.src_data_path)
            da_pred["x"] = xr.DataArray(
                da_src.x[da_pred.i0], dims=("i0",), attrs=dict(units=da_src["x"].units)
            )
            # OBS: j-indexing is positive "down" (from top-left corner) whereas
            # y-indexing is positive "up", so we have to reverse y here before
            # slicing
            da_pred["y"] = xr.DataArray(
                da_src.y[::-1][da_pred.j0],
                dims=("j0",),
                attrs=dict(units=da_src["y"].units),
            )
            da_pred = da_pred.swap_dims(dict(i0="x", j0="y")).sortby(["x", "y"])
            da_pred.attrs["lx_tile"] = float(da_src.x[N_tile[0]] - da_src.x[0])
            da_pred.attrs["ly_tile"] = float(da_src.y[N_tile[1]] - da_src.y[0])

            # make sure the lat lon coords are available later
            da_pred.coords["lat"] = da_src.lat.sel(x=da_pred.x, y=da_pred.y)
            da_pred.coords["lon"] = da_src.lon.sel(x=da_pred.x, y=da_pred.y)

        da_pred.attrs["model_path"] = self.model_path
        da_pred.attrs["image_path"] = self.image_path
        da_pred.attrs["src_data_path"] = self.src_data_path

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_pred.to_netcdf(self.output().fn)

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        fn_out = image_fn.replace(
            ".png", ".embeddings.{}_step.nc".format(self.step_size)
        )
        return XArrayTarget(str(image_path / fn_out))


class DatasetImagePredictionMapData(ImagePredictionMapData):
    """
    Create sliding-window tiling embeddings using a specific trained model for
    a specific scene of the dataset in `data_path`
    """

    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return SceneRegriddedData(data_path=self.data_path, scene_id=self.scene_id)

    @property
    def image_path(self):
        return self.input()["image"].fn

    @property
    def src_data_path(self):
        return self.input()["data"].fn

    def output(self):
        fn = "{}.embeddings.{}_step.nc".format(self.scene_id, self.step_size)
        model_name = Path(self.model_path).name.replace(".pkl", "").replace(".ckpt", "")
        p_out = Path(self.data_path) / "embeddings" / "rect" / model_name / fn
        return XArrayTarget(str(p_out))


class FullDatasetImagePredictionMapData(SceneBulkProcessingBaseTask):
    """
    Create sliding-window tiling embeddings using a specific trained model for
    all scenes in the dataset in `data_path`
    """

    model_path = luigi.Parameter()
    step_size = luigi.Parameter()

    TaskClass = DatasetImagePredictionMapData

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            model_path=self.model_path,
            step_size=self.step_size,
        )


class ImagePredictionMapImageTiles(luigi.Task):
    """
    Generate image files for the tiles sampled from a larger rect domain
    """

    image_path = luigi.Parameter()
    src_data_path = luigi.OptionalParameter()
    step_size = luigi.IntParameter(default=10)

    def run(self):
        raise NotImplementedError("changed dataloader")
        img = Image.open(self.image_path)

        N_tile = (256, 256)
        tile_dataset = MovingWindowImageTilingDataset(
            img=img,
            transform=None,
            step=(self.step_size, self.step_size),
            N_tile=N_tile,
        )
        for n in range(len(tile_dataset)):
            tile_img = tile_dataset.get_image(n)
            i, j = tile_dataset.spatial_index_to_img_ij(n)
            filename = IMAGE_TILE_FILENAME_FORMAT.format(i0=i, j0=j)
            tile_filepath = Path(self.output().fn).parent / filename
            tile_img.save(str(tile_filepath))

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        tile_path = Path(image_path) / "tiles" / image_fn.replace(".png", "")
        tile_path.mkdir(exist_ok=True, parents=True)

        fn_tile00 = IMAGE_TILE_FILENAME_FORMAT.format(i=0, j=0)
        p = tile_path / fn_tile00
        return luigi.LocalTarget(str(p))


class DatasetImagePredictionMapImageTiles(ImagePredictionMapImageTiles):
    """
    Generate image files for the tiles sampled from a larger rect domain for a
    specific scene from a dataset
    """

    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return SceneRegriddedData(data_path=self.data_path, scene_id=self.scene_id)

    @property
    def image_path(self):
        return self.input()["image"].fn

    @property
    def src_data_path(self):
        return self.input()["data"].fn

    def output(self):
        dir_name = "{}_tiles_{}step".format(self.scene_id, self.step_size)
        p_out = Path(self.data_path) / "composites" / "rect" / "tiles" / dir_name
        p_out.mkdir(exist_ok=True, parents=True)
        # TODO: fix this, we should actually be working out how many tiles to
        # produce from the source dataset
        fn_tile00 = IMAGE_TILE_FILENAME_FORMAT.format(i0=0, j0=0)
        p = p_out / fn_tile00
        return luigi.LocalTarget(str(p))


class FullDatasetImagePredictionMapImageTiles(SceneBulkProcessingBaseTask):
    step_size = luigi.Parameter()
    TaskClass = DatasetImagePredictionMapImageTiles

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            step_size=self.step_size,
        )


class AggregateFullDatasetImagePredictionMapData(luigi.Task):
    data_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()
    generate_tile_images = luigi.BoolParameter(default=False)

    def requires(self):
        reqs = {}
        reqs["data"] = FullDatasetImagePredictionMapData(
            data_path=self.data_path,
            model_path=self.model_path,
            step_size=self.step_size,
        )
        if self.generate_tile_images:
            reqs["images"] = FullDatasetImagePredictionMapImageTiles(
                data_path=self.data_path,
                step_size=self.step_size,
            )
        return reqs

    @property
    def scene_resolution(self):
        data_source = DataSource.load(path=self.data_path)
        if (
            "resolution" not in data_source.sampling
            or data_source.sampling.get("resolution") is None
        ):
            raise Exception(
                "To produce isometric grid resampling of the source data please "
                "define the grid-spacing by defining `resolution` (in meters/pixel) "
                "in the `sampling` part of the data source meta information"
            )
        return data_source.sampling["resolution"]

    def run(self):
        das = []
        for scene_id, input in self.input()["data"].items():
            da = input.open()
            da["scene_id"] = scene_id

            # turn attributes we want to keep into extra coordinates so that
            # we'll have them later
            for v in ["src_data_path", "image_path"]:
                if v in da.attrs:
                    value = da.attrs.pop(v)
                    da[v] = value
            das.append(da)

        da_all = xr.concat(das, dim="scene_id")
        da_all.name = "emb"
        da_all.attrs["step_size"] = self.step_size
        da_all.attrs["model_path"] = self.model_path

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_all.to_netcdf(self.output().fn)

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "").replace(".ckpt", "")
        fn = "all_embeddings.{}_step.nc".format(self.step_size)
        p = Path(self.data_path) / "embeddings" / "rect" / model_name / fn
        return XArrayTarget(str(p))
