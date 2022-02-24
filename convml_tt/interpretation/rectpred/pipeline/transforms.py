"""
luigi Tasks for putting embedding predictions on a rectangular domain through
transforms on the embedding dimensions

NB: currently untested
"""
from pathlib import Path

import joblib
import luigi
import xarray as xr

from ....pipeline import XArrayTarget
from ...embedding_transforms import apply_transform
from .data import AggregateFullDatasetImagePredictionMapData


class EmbeddingTransform(luigi.Task):
    """
    Apply a transform to the embeddings stored in `emb_input_path` of type
    `transform_type` (for example `pca`). You can optionally provide the "name"
    of the pretrained transform model (this is expected to be stored in the
    same directory as the embeddings, and then name will be parsed as
    `{name}.joblib`)
    """

    emb_input_path = luigi.Parameter()
    transform_type = luigi.Parameter()
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    transform_extra_args = luigi.OptionalParameter(default="")

    def _load_pretrained_transform_model(self):
        model_filename_format = "{}.model.joblib"
        model_filename = model_filename_format.format(self.pretrained_transform_model)
        models_path = Path(self._get_pretrained_transform_model_path())
        model_path = models_path / model_filename
        if not model_path.exists():
            pretrained_model_files = models_path.glob(model_filename_format.format("*"))
            avail_models = {
                p.name.replace(".model.joblib", ""): p for p in pretrained_model_files
            }
            avail_models_str = "\n".join(
                [f"\t{k: <20}: {v}" for (k, v) in avail_models.items()]
            )
            raise Exception(
                f"Couldn't find pre-trained transform model in `{model_path}`. "
                f"Available models:\n{avail_models_str}"
            )
        return joblib.load(str(model_path))

    def run(self):
        da_emb = xr.open_dataarray(self.emb_input_path)

        if self.pretrained_transform_model:
            pretrained_transform_model = self._load_pretrained_transform_model()
        else:
            pretrained_transform_model = None

        da_cluster, model = apply_transform(
            da=da_emb,
            transform_type=self.transform_type,
            pretrained_model=pretrained_transform_model,
            return_model=True,
            **self._parse_transform_extra_kwargs(),
        )

        if model is not None and self.pretrained_transform_model is None:
            joblib.dump(model, self.output()["model"].fn)

        da_cluster.attrs.update(da_emb.attrs)

        da_cluster.attrs.update(da_emb.attrs)
        da_cluster.name = "emb"
        da_cluster["i0"] = da_emb.i0
        da_cluster["j0"] = da_emb.j0
        if "lat" in da_emb.coords:
            da_cluster["lat"] = da_emb.coords["lat"]
        if "lon" in da_emb.coords:
            da_cluster["lon"] = da_emb.coords["lon"]
        da_cluster.attrs["transform_type"] = self.transform_type
        if self.transform_extra_args:
            da_cluster.attrs["transform_extra_args"] = self.transform_extra_args

        p_out = Path(self.output()["transformed_data"].fn).parent
        p_out.mkdir(exist_ok=True, parents=True)
        da_cluster.to_netcdf(self.output()["transformed_data"].fn)

    def _parse_transform_extra_kwargs(self):
        kwargs = {}
        if self.transform_extra_args:
            for s in self.transform_extra_args.split(","):
                k, v = s.split("=")
                if k in [
                    "min_cluster_size",
                    "min_samples",
                    "pca__n_components",
                    "n_components",
                ]:
                    v = int(v)
                else:
                    v = float(v)
                kwargs[k] = v
        return kwargs

    def _make_transform_model_filename(self):
        return f"{self._build_transform_identifier()}.model.joblib"

    def _get_pretrained_transform_model_path(self):
        """Return path to where pretrained transform models are expected to
        reside"""
        return Path(self.emb_input_path).parent

    def _build_transform_identifier(self):
        if self.pretrained_transform_model:
            return self.pretrained_transform_model

        s = f"{self.transform_type}_transform"
        if self.transform_extra_args:
            s += "__" + self.transform_extra_args.replace(",", "__").replace("=", "_")
        return s

    @property
    def _output_path(self):
        src_fullpath = Path(self.emb_input_path)
        src_path = src_fullpath.parent
        return src_path

    def output(self):
        src_fn = Path(self.emb_input_path).name

        output_path = self._output_path
        fn_data = src_fn.replace(".nc", f".{self._build_transform_identifier()}.nc")

        output = dict(transformed_data=XArrayTarget(str(output_path / fn_data)))
        if not self.pretrained_transform_model:
            # we only save the model when not using a pretrained transform model
            fn_model = self._make_transform_model_filename()
            output["model"] = luigi.LocalTarget(str(output_path / fn_model))

        return output


class DatasetEmbeddingTransform(EmbeddingTransform):
    """
    Create a netCDF file for the transformed embeddings of a single scene
    """

    data_path = luigi.OptionalParameter(default=".")
    model_path = luigi.Parameter()
    step_size = luigi.IntParameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return AggregateFullDatasetImagePredictionMapData(
            data_path=self.data_path,
            step_size=self.step_size,
            model_path=self.model_path,
        )

    def run(self):
        parent_output = yield EmbeddingTransform(
            emb_input_path=self.emb_input_path,
            transform_type=self.transform_type,
            pretrained_transform_model=self.pretrained_transform_model,
            transform_extra_args=self.transform_extra_args,
        )

        da_emb_all = parent_output["transformed_data"].open()
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_emb = da_emb_all.sel(scene_id=self.scene_id)

        # turn values that were previously attributes (but have been stored
        # as extra coordinates when we stacked) as attributes again
        for c in ["image_path", "src_data_path"]:
            value = da_emb[c].item()
            da_emb = da_emb.reset_coords(c, drop=True)
            da_emb.attrs[c] = value
        # not sure why this is necessary, but for some reason xarray doesn't
        # overwrite this variable and then thinks the old coordinates still
        # exist when we load the file again
        del da_emb.encoding["coordinates"]

        da_emb.to_netcdf(self.output().fn)

    @property
    def emb_input_path(self):
        return self.input().fn


class CreateAllPredictionMapsDataTransformed(EmbeddingTransform):
    data_path = luigi.OptionalParameter(default=".")
    embedding_model_path = luigi.Parameter()
    step_size = luigi.IntParameter()

    def requires(self):
        return AggregateFullDatasetImagePredictionMapData(
            data_path=self.data_path,
            model_path=self.embedding_model_path,
            step_size=self.step_size,
        )

    @property
    def emb_input_path(self):
        return self.input().fn
