from pathlib import Path

import luigi
import numpy as np
import xarray as xr

from fastai.basic_train import load_learner
from fastai.vision import open_image

from ...data.sources.satellite.rectpred import MakeRectRGBImage
from ...pipeline import XArrayTarget
from ...data.dataset import SceneBulkProcessingBaseTask
from ...architectures.triplet_trainer import monkey_patch_fastai

monkey_patch_fastai()  # noqa

IMAGE_TILE_FILENAME_FORMAT = "{i0:05d}_{j0:05d}.png"

N_TILE = (256, 256)


class FakeImagesList(list):
    def __init__(self, src_path, id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id
        self.src_path = src_path

    @property
    def size(self):
        return self[0].size

    def apply_tfms(self, tfms, **kwargs):
        items = []
        for item in self:
            items.append(item.apply_tfms(tfms, **kwargs))
        return items


class RectTiler:
    def __init__(self, img, N_tile, step):
        self.img = img
        self.ny, self.nx = img.size
        self.nxt, self.nyt = N_tile
        self.x_step, self.y_step = step

        self.i_ = np.arange(0, self.nx - self.nxt + 1, self.x_step)
        self.j_ = np.arange(0, self.ny - self.nyt + 1, self.y_step)

    @staticmethod
    def _crop_fastai_im(img, i, j, nx=256, ny=256):
        img_copy = img.__class__(img._px[:, j : j + ny, i : i + nx])
        # From PIL docs: The crop rectangle, as a (left, upper, right, lower)-tuple.
        return img_copy

    def get_tile_images(self):
        for i in self.i_:
            for j in self.j_:
                tile_img = self._crop_fastai_im(
                    self.img, i=i, j=j, nx=self.nxt, ny=self.nyt
                )
                yield (i, j), tile_img

    def make_tile_predictions(self, model):
        """
        Produce moving-window prediction array from `img` with `model` with
        step-size defined by `step` and tile-size `N_tile`.

        NB: j-indexing is from "top_left" i.e. is likely in the opposite order to
        what would be expected for y-axis of original image (positive being up)

        i0 and j0 coordinates denote center of each prediction tile
        """
        il = FakeImagesList(None, None)

        for tile_img in self.get_tile_images():
            il.append(tile_img)

        final_shape = (len(self.i_), len(self.j_), -1)
        result = np.stack(model.predict(il)[1]).reshape(final_shape)

        return xr.DataArray(
            result,
            dims=("i0", "j0", "emb_dim"),
            coords=dict(i0=self.i_ + self.nxt // 2, j0=self.j_ + self.nyt // 2),
            attrs=dict(tile_nx=self.nxt, tile_ny=self.nyt),
        )


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

    def run(self):
        model_fullpath = Path(self.model_path)
        model_path, model_fn = model_fullpath.parent, model_fullpath.name
        model = load_learner(model_path, model_fn)
        img = open_image(self.image_path)

        tiler = RectTiler(img=img, N_tile=n_TILE, step=(self.step_size, self.step_size))
        da_pred = tiler.make_tile_predictions(model=model)

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

        da_pred.to_netcdf(self.output().fn)

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        fn_out = image_fn.replace(
            ".png", ".embeddings.{}_step.nc".format(self.step_size)
        )
        return XArrayTarget(str(image_path / fn_out))


class ImagePredictionMapImageTiles(luigi.Task):
    image_path = luigi.Parameter()
    src_data_path = luigi.OptionalParameter()
    step_size = luigi.IntParameter(default=10)

    def run(self):
        img = open_image(self.image_path)

        N_tile = (256, 256)

        tiler = RectTiler(img=img, N_tile=N_tile, step=(self.step_size, self.step_size))

        for (i, j), tile_img in tiler.get_tile_images():
            nxt, nyt = tiler.nxt, tiler.nyt
            filename = IMAGE_TILE_FILENAME_FORMAT.format(
                i0=i + nxt // 2, j0=j + nyt // 2
            )
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
    dataset_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return MakeRectRGBImage(dataset_path=self.dataset_path, scene_id=self.scene_id)

    @property
    def image_path(self):
        return self.input().fn

    @property
    def src_data_path(self):
        return self.requires().input().fn

    def output(self):
        dir_name = "{}_tiles_{}step".format(self.scene_id, self.step_size)
        p_out = Path(self.dataset_path) / "composites" / "rect" / "tiles" / dir_name
        p_out.mkdir(exist_ok=True, parents=True)
        fn_tile00 = IMAGE_TILE_FILENAME_FORMAT.format(i0=128, j0=128)
        p = p_out / fn_tile00
        return luigi.LocalTarget(str(p))


class AllDatasetImagePredictionMapImageTiles(SceneBulkProcessingBaseTask):
    step_size = luigi.Parameter()

    TaskClass = DatasetImagePredictionMapImageTiles

    def _get_task_class_kwargs(self):
        return dict(
            step_size=self.step_size,
        )


class DatasetImagePredictionMapData(ImagePredictionMapData):
    dataset_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return MakeRectRGBImage(dataset_path=self.dataset_path, scene_id=self.scene_id)

    @property
    def image_path(self):
        return self.input().fn

    @property
    def src_data_path(self):
        return self.requires().input().fn

    def output(self):
        fn = "{}.embeddings.{}_step.nc".format(self.scene_id, self.step_size)
        p_out = Path(self.dataset_path) / "composites" / "rect" / fn
        return XArrayTarget(str(p_out))


class FullDatasetImagePredictionMapData(SceneBulkProcessingBaseTask):
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()

    TaskClass = DatasetImagePredictionMapData

    def _get_task_class_kwargs(self):
        return dict(
            model_path=self.model_path,
            step_size=self.step_size,
        )


class AggregateFullDatasetImagePredictionMapData(luigi.Task):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()

    def requires(self):
        return FullDatasetImagePredictionMapData(
            dataset_path=self.dataset_path,
            model_path=self.model_path,
            step_size=self.step_size,
        )

    def run(self):
        das = []
        for scene_id, input in self.input().items():
            da = input.open()
            da["scene_id"] = scene_id
            das.append(da)

        da_all = xr.concat(das, dim="scene_id")
        da_all.name = "emb"
        da_all.attrs["step_size"] = self.step_size
        da_all.attrs["model_path"] = self.model_path

        # remove attributes that only applies to one source image
        if "src_data_path" in da_all.attrs:
            del da_all.attrs["src_data_path"]
        del da_all.attrs["image_path"]

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_all.to_netcdf(self.output().fn)

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "")
        fn = "all_embeddings.{}_step.nc".format(self.step_size)
        p = Path(self.dataset_path) / "embeddings" / "rect" / model_name / fn
        return XArrayTarget(str(p))
