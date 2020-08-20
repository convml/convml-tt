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


def crop_fastai_im(img, i, j, nx=256, ny=256):
    img_copy = img.__class__(img._px[:, j : j + ny, i : i + nx])
    # From PIL docs: The crop rectangle, as a (left, upper, right, lower)-tuple.
    return img_copy


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


def rect_predict(model, img, N_tile, step=(10, 10)):
    """
    Produce moving-window prediction array from `img` with `model` with
    step-size defined by `step` and tile-size `N_tile`.

    NB: j-indexing is from "top_left" i.e. is likely in the opposite order to
    what would be expected for y-axis of original image (positive being up)

    i0 and j0 coordinates denote center of each prediction tile
    """
    ny, nx = img.size
    nxt, nyt = N_tile
    x_step, y_step = step

    il = FakeImagesList(None, None)
    i_ = np.arange(0, nx - nxt + 1, x_step)
    j_ = np.arange(0, ny - nyt + 1, y_step)

    for n in i_:
        for m in j_:
            il.append(crop_fastai_im(img, n, m, nx=nxt, ny=nyt))

    result = np.stack(model.predict(il)[1]).reshape((len(i_), len(j_), -1))

    da = xr.DataArray(
        result,
        dims=("i0", "j0", "emb_dim"),
        coords=dict(i0=i_ + nxt // 2, j0=j_ + nyt // 2),
        attrs=dict(tile_nx=nxt, tile_ny=nyt),
    )

    return da


class ImagePredictionMapData(luigi.Task):
    model_path = luigi.Parameter()
    image_path = luigi.Parameter()
    src_data_path = luigi.OptionalParameter()
    step_size = luigi.IntParameter(default=10)

    def run(self):
        model_fullpath = Path(self.model_path)
        model_path, model_fn = model_fullpath.parent, model_fullpath.name
        model = load_learner(model_path, model_fn)
        img = open_image(self.image_path)

        N_tile = (256, 256)

        da_pred = rect_predict(
            model=model, img=img, step=(self.step_size, self.step_size), N_tile=N_tile
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
        return dict(model_path=self.model_path, step_size=self.step_size,)


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
