"""
luigi Tasks for running plotting pipeline on rectangular domain datasets
"""
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import xarray as xr
from convml_data import DataSource
from convml_data.pipeline import SceneBulkProcessingBaseTask

from ....pipeline import XArrayTarget
from ..plot import make_components_annotation_map_image, make_rgb_annotation_map_image
from .data import DatasetImagePredictionMapData
from .transforms import DatasetEmbeddingTransform


class ComponentsAnnotationMapImage(luigi.Task):
    input_path = luigi.Parameter()
    components = luigi.ListParameter(default=[0, 1, 2])
    src_data_path = luigi.Parameter()
    col_wrap = luigi.IntParameter(default=2)

    def run(self):
        da_emb = xr.open_dataarray(self.input_path)

        fig, axes = make_components_annotation_map_image(
            da_emb=da_emb, components=self.components, col_wrap=self.col_wrap
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
    data_path = luigi.Parameter()
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
                data_path=self.data_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
            )
        else:
            return DatasetEmbeddingTransform(
                data_path=self.data_path,
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
        return self.data_path

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "")

        fn = "{}.{}_step.{}_transform.map.{}__comp.png".format(
            self.scene_id,
            self.step_size,
            self.transform_type,
            "_".join([str(v) for v in self.components]),
        )

        p_root = Path(self.data_path) / "embeddings" / "rect" / model_name

        if self.pretrained_transform_model is not None:
            p = p_root / self.pretrained_transform_model / "components_map" / fn
        else:
            p = p_root / "components_map" / fn
        return XArrayTarget(str(p))


class AllDatasetComponentAnnotationMapImages(SceneBulkProcessingBaseTask):
    model_path = luigi.Parameter()
    step_size = luigi.IntParameter()
    transform_type = luigi.OptionalParameter()
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    components = luigi.ListParameter(default=[0, 1, 2])

    TaskClass = DatasetComponentsAnnotationMapImage

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            model_path=self.model_path,
            step_size=self.step_size,
            transform_type=self.transform_type,
            pretrained_transform_model=self.pretrained_transform_model,
            components=self.components,
        )


class RGBAnnotationMapImage(luigi.Task):
    """
    Create RGB overlay plot from embeddings on Cartesian domain reading the
    embeddings from `input_path`
    """

    input_path = luigi.Parameter()
    rgb_components = luigi.ListParameter(default=[0, 1, 2])
    src_data_path = luigi.Parameter()
    render_tiles = luigi.BoolParameter(default=False)

    def make_plot(self, da_emb):
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            return make_rgb_annotation_map_image(
                da_emb=da_emb,
                rgb_components=self.rgb_components,
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
    data_path = luigi.Parameter()
    step_size = luigi.IntParameter()
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
                data_path=self.data_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
            )
        else:
            return DatasetEmbeddingTransform(
                data_path=self.data_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
                transform_type=self.transform_type,
                transform_extra_args=self.transform_extra_args,
                pretrained_model=self.pretrained_transform_model,
            )

    def run(self):
        datasource = DataSource.load(self.data_path)
        da_emb = xr.open_dataarray(self.input_path)

        dx = da_emb.lx_tile / da_emb.tile_nx
        dy = da_emb.ly_tile / da_emb.tile_ny

        assert dx == dy

        tile_resolution = dx / 1000.0
        step_km = self.step_size * tile_resolution
        domain = datasource.domain
        lat0, lon0 = domain.central_latitude, domain.central_longitude
        model_name = self.model_path.replace(".pkl", "")

        title_parts = [
            self.scene_id,
            f"(lat0, lon0)=({lat0}, {lon0})",
            (
                f"{model_name} NN model, {da_emb.tile_nx} x {da_emb.tile_ny} tiles "
                f"at {tile_resolution:.2f}km resolution, {da_emb.step_size} step ({step_km:.2f}km)"
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
        return self.data_path

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

        p_root = Path(self.data_path) / "embeddings" / "rect" / model_name
        if self.pretrained_transform_model is not None:
            p = p_root / self.pretrained_transform_model / fn
        else:
            p = p_root / fn
        return XArrayTarget(str(p))


class AllDatasetRGBAnnotationMapImages(SceneBulkProcessingBaseTask):
    model_path = luigi.Parameter()
    step_size = luigi.IntParameter()
    transform_type = luigi.OptionalParameter()
    transform_extra_args = luigi.OptionalParameter(default=None)
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    rgb_components = luigi.ListParameter(default=[0, 1, 2])

    TaskClass = DatasetRGBAnnotationMapImage

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            model_path=self.model_path,
            step_size=self.step_size,
            transform_type=self.transform_type,
            transform_extra_args=self.transform_extra_args,
            pretrained_transform_model=self.pretrained_transform_model,
            rgb_components=self.rgb_components,
        )
