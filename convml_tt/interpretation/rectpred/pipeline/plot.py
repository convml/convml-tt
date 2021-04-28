"""
luigi Tasks for running plotting pipeline on rectangular domain datasets
"""
from pathlib import Path

import luigi
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ....data.dataset import SceneBulkProcessingBaseTask, TripletDataset
from ....data.sources.satellite import tiler
from ....data.sources.satellite.rectpred import MakeRectRGBImage
from ....pipeline import XArrayTarget
from .data import DatasetImagePredictionMapData
from .transform import DatasetEmbeddingTransform


class ComponentsAnnotationMapImage(luigi.Task):
    input_path = luigi.Parameter()
    components = luigi.ListParameter(default=[0, 1, 2])
    src_data_path = luigi.Parameter()
    col_wrap = luigi.IntParameter(default=2)

    def run(self):
        da_emb = xr.open_dataarray(self.input_path)

        da_emb.coords["pca_dim"] = np.arange(da_emb.pca_dim.count())

        da_emb = da_emb.assign_coords(
            x=da_emb.x / 1000.0,
            y=da_emb.y / 1000.0,
            explained_variance=np.round(da_emb.explained_variance, 2),
        )
        da_emb.x.attrs["units"] = "km"
        da_emb.y.attrs["units"] = "km"

        img, img_extent = self.get_image(da_emb=da_emb)

        img_extent = np.array(img_extent) / 1000.0

        # find non-xy dim
        d_not_xy = next(filter(lambda d: d not in ["x", "y"], da_emb.dims))

        N_subplots = len(self.components) + 1
        data_r = 3.0
        ncols = self.col_wrap
        size = 3.0

        nrows = int(np.ceil(N_subplots / ncols))
        figsize = (int(size * data_r * ncols), int(size * nrows))

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=nrows,
            ncols=ncols,
            subplot_kw=dict(aspect=1),
            sharex=True,
        )

        ax = axes.flatten()[0]
        ax.imshow(img, extent=img_extent)
        ax.set_title(da_emb.scene_id.item())

        for n, ax in zip(self.components, axes.flatten()[1:]):
            ax.imshow(img, extent=img_extent)
            da_ = da_emb.sel(**{d_not_xy: n})
            da_ = da_.drop(["i0", "j0", "scene_id"])

            da_.plot.imshow(ax=ax, y="y", alpha=0.5, add_colorbar=False)

            ax.set_xlim(*img_extent[:2])
            ax.set_ylim(*img_extent[2:])

        [ax.set_aspect(1) for ax in axes.flatten()]
        [ax.set_xlabel("") for ax in axes[:-1, :].flatten()]

        plt.tight_layout()

        fig.text(
            0.0,
            -0.02,
            "cum. explained variance: {}".format(
                np.cumsum(da_emb.explained_variance.values)
            ),
        )

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(self.output().fn, bbox_inches="tight")
        plt.close(fig)

    def get_image(self, da_emb):
        raise NotImplementedError

    def output(self):
        image_fullpath = Path(self.input_path)
        src_path, src_fn = image_fullpath.parent, image_fullpath.name

        fn_out = src_fn.replace(
            ".nc",
            ".map.{}__comp.png".format("_".join([str(v) for v in self.components])),
        )

        p = Path(src_path) / fn_out

        return luigi.LocalTarget(str(p))


class DatasetComponentsAnnotationMapImage(ComponentsAnnotationMapImage):
    dataset_path = luigi.Parameter()
    step_size = luigi.Parameter()
    model_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    transform_type = luigi.OptionalParameter()
    transform_extra_args = luigi.OptionalParameter()
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    components = luigi.ListParameter(default=[0, 1, 2])
    crop_img = luigi.BoolParameter(default=False)

    def requires(self):
        if self.transform_type is None:
            return DatasetImagePredictionMapData(
                dataset_path=self.dataset_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
            )
        else:
            return DatasetEmbeddingTransform(
                dataset_path=self.dataset_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
                transform_type=self.transform_type,
                transform_extra_args=self.transform_extra_args,
                pretrained_model=self.pretrained_transform_model,
            )

    @property
    def input_path(self):
        return self.input().fn

    @property
    def src_data_path(self):
        return self.dataset_path

    def get_image(self, da_emb):
        img_path = (
            MakeRectRGBImage(dataset_path=self.dataset_path, scene_id=self.scene_id)
            .output()
            .fn
        )

        if self.crop_img:
            return _get_img_with_extent_cropped(da_emb, img_path)
        else:
            return _get_img_with_extent(
                da_emb=da_emb, img_fn=img_path, dataset_path=self.dataset_path
            )

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "")

        fn = "{}.{}_step.{}_transform.map.{}__comp.png".format(
            self.scene_id,
            self.step_size,
            self.transform_type,
            "_".join([str(v) for v in self.components]),
        )

        p_root = Path(self.dataset_path) / "embeddings" / "rect" / model_name

        if self.pretrained_transform_model is not None:
            p = p_root / self.pretrained_transform_model / "components_map" / fn
        else:
            p = p_root / "components_map" / fn
        return XArrayTarget(str(p))


class AllDatasetComponentAnnotationMapImages(SceneBulkProcessingBaseTask):
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()
    transform_type = luigi.OptionalParameter()
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    components = luigi.ListParameter(default=[0, 1, 2])

    TaskClass = DatasetComponentsAnnotationMapImage

    def _get_task_class_kwargs(self):
        return dict(
            model_path=self.model_path,
            step_size=self.step_size,
            transform_type=self.transform_type,
            pretrained_transform_model=self.pretrained_transform_model,
            components=self.components,
        )


class RGBAnnotationMapImage(luigi.Task):
    input_path = luigi.Parameter()
    rgb_components = luigi.ListParameter(default=[0, 1, 2])
    src_data_path = luigi.Parameter()
    render_tiles = luigi.BoolParameter(default=False)

    def make_plot(self, da_emb):
        return make_rgb_annotation_map_image(
            da=da_emb,
            rgb_components=self.rgb_components,
            dataset_path=self.dataset_path,
        )

    def run(self):
        da_emb = xr.open_dataarray(self.input_path)
        fig, axes = self.make_plot(da_emb=da_emb)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(self.output().fn, fig=fig, bbox_inches="tight")

    def output(self):
        image_fullpath = Path(self.input_path)
        src_path, src_fn = image_fullpath.parent, image_fullpath.name

        fn_out = src_fn.replace(
            ".nc",
            ".rgb_map.{}__comp.png".format(
                "_".join([str(v) for v in self.rgb_components])
            ),
        )

        p = Path(src_path) / fn_out

        return luigi.LocalTarget(str(p))


class DatasetRGBAnnotationMapImage(RGBAnnotationMapImage):
    dataset_path = luigi.Parameter()
    step_size = luigi.Parameter()
    model_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    transform_type = luigi.OptionalParameter(default=None)
    transform_extra_args = luigi.OptionalParameter(default=None)
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    rgb_components = luigi.ListParameter(default=[0, 1, 2])
    crop_img = luigi.BoolParameter(default=False)

    def requires(self):
        if self.transform_type is None:
            return DatasetImagePredictionMapData(
                dataset_path=self.dataset_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
            )
        else:
            return DatasetEmbeddingTransform(
                dataset_path=self.dataset_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
                transform_type=self.transform_type,
                transform_extra_args=self.transform_extra_args,
                pretrained_model=self.pretrained_transform_model,
                # n_clusters=max(self.rgb_components)+1, TODO: put into transform_extra_args
            )

    def run(self):
        dataset = TripletDataset.load(self.dataset_path)
        da_emb = xr.open_dataarray(self.input_path)

        N_tile = (256, 256)
        model_resolution = da_emb.lx_tile / N_tile[0] / 1000.0
        domain_rect = dataset.extra["rectpred"]["domain"]
        lat0, lon0 = domain_rect["lat0"], domain_rect["lon0"]

        title_parts = [
            self.scene_id,
            "(lat0, lon0)=({}, {})".format(lat0, lon0),
            "{} NN model, {} x {} tiles at {:.2f}km resolution".format(
                self.model_path.replace(".pkl", ""),
                N_tile[0],
                N_tile[1],
                model_resolution,
            ),
            "prediction RGB from {} components [{}]".format(
                da_emb.transform_type, ", ".join([str(v) for v in self.rgb_components])
            ),
        ]
        if self.transform_extra_args:
            title_parts.append(self.requires()._build_transform_identifier())
        title = "\n".join(title_parts)

        fig, axes = self.make_plot(da_emb=da_emb)
        fig.suptitle(title, y=1.05)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(self.output().fn, fig=fig, bbox_inches="tight")

    @property
    def input_path(self):
        return self.input().fn

    @property
    def src_data_path(self):
        return self.dataset_path

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "")

        fn_parts = [
            self.scene_id,
            f"{self.step_size}_step",
            "rgb_map",
            "{}__comp".format("_".join([str(v) for v in self.rgb_components])),
        ]

        if self.transform_type:
            fn_parts.insert(2, self.requires()._build_transform_identifier())

        fn = f"{'.'.join(fn_parts)}.png"

        p_root = Path(self.dataset_path) / "embeddings" / "rect" / model_name
        if self.pretrained_transform_model is not None:
            p = p_root / self.pretrained_transform_model / fn
        else:
            p = p_root / fn
        return XArrayTarget(str(p))


class AllDatasetRGBAnnotationMapImages(SceneBulkProcessingBaseTask):
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()
    transform_type = luigi.OptionalParameter()
    transform_extra_args = luigi.OptionalParameter(default=None)
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    rgb_components = luigi.ListParameter(default=[0, 1, 2])

    TaskClass = DatasetRGBAnnotationMapImage

    def _get_task_class_kwargs(self):
        return dict(
            model_path=self.model_path,
            step_size=self.step_size,
            transform_type=self.transform_type,
            transform_extra_args=self.transform_extra_args,
            pretrained_transform_model=self.pretrained_transform_model,
            rgb_components=self.rgb_components,
        )
