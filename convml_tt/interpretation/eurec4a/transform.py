from pathlib import Path

import xarray as xr
import luigi
import sklearn.cluster
import hdbscan
import joblib

from ...pipeline import XArrayTarget
from .data import AggregateFullDatasetImagePredictionMapData


def _apply_transform_function(da, fn, transform_name):
    # stack all other dims apart from the `emb_dim`
    dims = list(da.dims)
    if "emb_dim" in dims:
        dims.remove("emb_dim")
    elif "pca_dim" in dims:
        dims.remove("pca_dim")
    else:
        raise NotImplementedError(da.dims)
    da_stacked = da.stack(dict(n=dims))
    arr = fn(X=da_stacked.T)
    if len(arr.shape) == 2:
        dims = ("n", "{}_dim".format(transform_name))
    else:
        dims = ("n",)
    return xr.DataArray(arr, dims=dims, coords=dict(n=da_stacked.n)).unstack("n")


def apply_transform(da, transform_type, pretrained_model=None, **kwargs):
    add_meta = None
    model = None

    if transform_type == "kmeans":
        fn_transform = sklearn.cluster.KMeans(**kwargs).fit_predict
    elif transform_type in ["pca", "pca_clipped"]:
        if pretrained_model is not None:
            model = pretrained_model
            fn_transform = model.transform
        else:
            model = sklearn.decomposition.PCA(**kwargs)
            fn_transform = model.fit_transform

        def add_meta(da):
            da["explained_variance"] = (
                "{}_dim".format(transform_type),
                model.explained_variance_ratio_,
            )

        if transform_type == "pca_clipped":
            da = da.isel(x=slice(1, -1), y=slice(1, -1))

    elif transform_type == "hdbscan":
        model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, **kwargs)

        def fn_transform(X):
            return model.fit(X).labels_

        def add_meta(da):
            return xr.DataArray(
                model.probabilities_,
                dims=("n"),
                coords=dict(n=da.stack(dict(n=da.dims)).n),
            ).unstack("n")

    elif transform_type == "pca_hdbscan":
        try:
            pca__n_components = kwargs.pop("pca__n_components")
        except KeyError:
            raise Exception(
                "To use HDBSCAN with PCA analysis first you need"
                " provide the number of PCA components to keep with"
                " the `pca__n_components` argument"
            )
        pca_model = sklearn.decomposition.PCA(n_components=pca__n_components)
        hdbscan_model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, **kwargs)
        model = hdbscan_model

        def fn_transform(X):
            X1 = pca_model.fit_transform(X)
            return hdbscan_model.fit(X1).labels_

        def add_meta(da):
            da.attrs["notes"] = "used PCA before HDBSCAN"

    else:
        raise NotImplementedError(transform_type)

    da_cluster = _apply_transform_function(
        da=da, fn=fn_transform, transform_name=transform_type
    )

    da_cluster.attrs.update(da.attrs)

    if add_meta is not None:
        add_meta(da_cluster)

    da_cluster.attrs.update(da.attrs)
    da_cluster.name = "emb"
    for v in ["i0", "j0"]:
        if v in da:
            da_cluster[v] = da[v]
    da_cluster.attrs["transform_type"] = transform_type
    if kwargs:
        s = ",".join([f"{k}={v}" for (k, v) in kwargs])
        da_cluster.attrs["transform_extra_args"] = s
    return da_cluster, model


class EmbeddingTransform(luigi.Task):
    input_path = luigi.Parameter()
    transform_type = luigi.Parameter()
    pretrained_model = luigi.OptionalParameter(default=None)
    transform_extra_args = luigi.OptionalParameter()

    def run(self):
        da_emb = xr.open_dataarray(self.input_path)

        if self.pretrained_model:
            p = Path(self._get_pretrained_model_path()) / "{}.joblib".format(
                self.pretrained_model
            )
            if not p.exists():
                raise Exception(
                    "Couldn't find pre-trained transform" " model in `{}`".format(p)
                )
            pretrained_model = joblib.load(str(p))
        else:
            pretrained_model = None

        da_cluster, model = apply_transform(
            da=da_emb,
            transform_type=self.transform_type,
            pretrained_model=pretrained_model,
            **self._parse_transform_extra_kwargs(),
        )

        if model is not None and self.pretrained_model is None:
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

    def _get_pretrained_model_path(self):
        """Return path to where pretrained transform models are expected to
        reside"""
        return Path(self.input_path).parent

    def _build_transform_identifier(self):
        s = f"{self.transform_type}_transform"
        if self.transform_extra_args:
            s += "__" + self.transform_extra_args.replace(",", "__").replace("=", "_")
        return s

    def output(self):
        src_fullpath = Path(self.input_path)
        src_path, src_fn = src_fullpath.parent, src_fullpath.name

        fn_data = src_fn.replace(".nc", f".{self._build_transform_identifier()}.nc")

        if self.pretrained_model:
            # we won't be resaving a pre-trained model, so there's only a
            # transformed data output, but in a subdir named after the
            # pretrained-model
            p = src_path / self.pretrained_model / fn_data
            return dict(transformed_data=XArrayTarget(str(p)))

        else:
            fn_model = self._make_transform_model_filename()
            return dict(
                model=luigi.LocalTarget(str(src_path / fn_model)),
                transformed_data=XArrayTarget(str(src_path / fn_data)),
            )


class DatasetEmbeddingTransform(EmbeddingTransform):
    """
    Create a netCDF file for the transformed embeddings of a single scene
    """

    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return AggregateFullDatasetImagePredictionMapData(
            dataset_path=self.dataset_path,
            step_size=self.step_size,
            model_path=self.model_path,
        )

    def run(self):
        parent_output = yield EmbeddingTransform(
            input_path=self.input_path,
            transform_type=self.transform_type,
            pretrained_model=self.pretrained_model,
            transform_extra_args=self.transform_extra_args,
        )

        import ipdb

        with ipdb.launch_ipdb_on_exception():
            da_emb_all = parent_output["transformed_data"].open()
            Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
            da_emb = da_emb_all.sel(scene_id=self.scene_id)
            da_emb.to_netcdf(self.output().fn)

    @property
    def input_path(self):
        return self.input().fn

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "")

        fn = "{}.{}_step.{}.nc".format(
            self.scene_id,
            self.step_size,
            self._build_transform_identifier(),
        )

        if self.pretrained_model is not None:
            name = self.pretrained_model
            p = Path(self.dataset_path) / "embeddings" / "rect" / model_name / name / fn
        else:
            p = Path(self.dataset_path) / "embeddings" / "rect" / model_name / fn

        return XArrayTarget(str(p))

    def _get_pretrained_model_path(self):
        """Return path to where pretrained transform models are expected to
        reside"""
        return Path(self.dataset_path) / "transform_models"


class CreateAllPredictionMapsDataTransformed(EmbeddingTransform):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()

    def requires(self):
        return AggregateFullDatasetImagePredictionMapData(
            dataset_path=self.dataset_path,
            model_path=self.model_path,
            step_size=self.step_size,
        )

    @property
    def input_path(self):
        return self.input().fn

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "")
        fn = "all_embeddings.{}_step.{}.nc".format(
            self.step_size,
            self._build_transform_identifier(),
        )
        p_root = (
            Path(self.dataset_path)
            / "embeddings"
            / "rect"
            / model_name
            / "components_map"
        )
        if self.pretrained_transform_model is not None:
            p = p_root / self.pretrained_transform_model / fn
        else:
            p = p_root / fn
        return XArrayTarget(str(p))
