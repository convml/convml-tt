from pathlib import Path

import hdbscan
import skimage.color
import luigi
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import joblib

from fastai.basic_train import load_learner
from fastai.vision import open_image

from convml_tt.architectures.triplet_trainer import monkey_patch_fastai
monkey_patch_fastai() # noqa

from convml_tt.data.sources.satellite.rectpred import MakeRectRGBImage
from ...pipeline import XArrayTarget
from ...data.dataset import SceneBulkProcessingBaseTask, TripletDataset
from ...data.sources.satellite import tiler


def crop_fastai_im(img, i, j, nx=256, ny=256):
    img_copy = img.__class__(img._px[:, j:j+ny, i:i+nx])
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
    i_ = np.arange(0, nx-nxt+1, x_step)
    j_ = np.arange(0, ny-nyt+1, y_step)

    for n in i_:
        for m in j_:
            il.append(crop_fastai_im(img, n, m, nx=nxt, ny=nyt))

    result = np.stack(model.predict(il)[1]).reshape((len(i_), len(j_), -1))

    da = xr.DataArray(
        result,
        dims=('i0', 'j0', 'emb_dim'),
        coords=dict(i0=i_ + nxt//2, j0=j_ + nyt//2),
        attrs=dict(tile_nx=nxt, tile_ny=nyt)
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
            model=model, img=img, step=(self.step_size, self.step_size),
            N_tile=N_tile
        )

        if self.src_data_path:
            da_src = xr.open_dataarray(self.src_data_path)
            da_pred['x'] = xr.DataArray(
                da_src.x[da_pred.i0], dims=('i0',),
                attrs=dict(units=da_src['x'].units)
            )
            # OBS: j-indexing is positive "down" (from top-left corner) whereas
            # y-indexing is positive "up", so we have to reverse y here before
            # slicing
            da_pred['y'] = xr.DataArray(
                da_src.y[::-1][da_pred.j0], dims=('j0',),
                attrs=dict(units=da_src['y'].units)
            )
            da_pred = da_pred.swap_dims(dict(i0='x', j0='y')).sortby(["x", "y"])
            da_pred.attrs['lx_tile'] = float(da_src.x[N_tile[0]]-da_src.x[0])
            da_pred.attrs['ly_tile'] = float(da_src.y[N_tile[1]]-da_src.y[0])

        da_pred.attrs['model_path'] = self.model_path
        da_pred.attrs['image_path'] = self.image_path
        da_pred.attrs['src_data_path'] = self.src_data_path

        da_pred.to_netcdf(self.output().fn)

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        fn_out = image_fn.replace(
            '.png', '.embeddings.{}_step.nc'.format(self.step_size))
        return XArrayTarget(str(image_path/fn_out))


class DatasetImagePredictionMapData(ImagePredictionMapData):
    dataset_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return MakeRectRGBImage(
            dataset_path=self.dataset_path,
            scene_id=self.scene_id
        )

    @property
    def image_path(self):
        return self.input().fn

    @property
    def src_data_path(self):
        return self.requires().input().fn

    def output(self):
        fn = "{}.embeddings.{}_step.nc".format(
            self.scene_id, self.step_size
        )
        p_out = (
            Path(self.dataset_path)/"composites"/"rect"/fn
        )
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
            step_size=self.step_size
        )

    def run(self):
        das = []
        for scene_id, input in self.input().items():
            da = input.open()
            da['scene_id'] = scene_id
            das.append(da)

        da_all = xr.concat(das, dim='scene_id')
        da_all.name = 'emb'
        da_all.attrs['step_size'] = self.step_size
        da_all.attrs['model_path'] = self.model_path

        # remove attributes that only applies to one source image
        if 'src_data_path' in da_all.attrs:
            del(da_all.attrs['src_data_path'])
        del(da_all.attrs['image_path'])

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_all.to_netcdf(self.output().fn)

    def output(self):
        model_name = Path(self.model_path).name.replace('.pkl', '')
        fn = "all_embeddings.{}_step.nc".format(self.step_size)
        p = Path(self.dataset_path)/"embeddings"/"rect"/model_name/fn
        return XArrayTarget(str(p))


class EmbeddingTransform(luigi.Task):
    input_path = luigi.Parameter()
    transform_type = luigi.Parameter()
    pretrained_model = luigi.OptionalParameter(default=None)
    transform_extra_args = luigi.OptionalParameter()

    def run(self):
        da_emb = xr.open_dataarray(self.input_path)

        add_meta = None
        model = None
        if self.transform_type == 'kmeans':
            fn_transform = sklearn.cluster.KMeans(
            ).fit_predict
        elif self.transform_type in ['pca', 'pca_clipped']:
            if self.pretrained_model is not None:
                p = Path(self._get_pretrained_model_path())/"{}.joblib".format(self.pretrained_model)
                if not p.exists():
                    raise Exception("Couldn't find pre-trained transform"
                                    " model in `{}`".format(p))
                model = joblib.load(str(p))
                fn_transform = model.transform
            else:
                model = sklearn.decomposition.PCA()
                fn_transform = model.fit_transform

            def add_meta(da):
                da['explained_variance'] = (
                    '{}_dim'.format(self.transform_type),
                    model.explained_variance_ratio_
                )
            if self.transform_type == 'pca_clipped':
                da_emb = da_emb.isel(x=slice(1, -1), y=slice(1, -1))

        elif self.transform_type == "hdbscan":
            kwargs = {}
            if self.transform_extra_args:
                for s in self.transform_extra_args.split(","):
                    k, v = s.split("=")
                    if k in ["min_cluster_size", "min_samples"]:
                        v = int(v)
                    else:
                        v = float(v)
                    kwargs[k] = v
            model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, **kwargs)
            fn_transform = lambda X: model.fit(X).labels_

            def add_meta(da):
                return xr.DataArray(
                    model.probabilities_,
                    dims=('n'),
                    coords=dict(n=da.stack(dict(n=da.dims)).n)
                ).unstack('n')
        else:
            raise NotImplementedError(self.transform_type)

        da_cluster = _apply_transform(
            da=da_emb, fn=fn_transform, transform_name=self.transform_type
        )

        if model is not None and self.pretrained_model is None:
            joblib.dump(model, self.output()['model'].fn)

        da_cluster.attrs.update(da_emb.attrs)

        if add_meta is not None:
            add_meta(da_cluster)

        da_cluster.attrs.update(da_emb.attrs)
        da_cluster.name = 'emb'
        da_cluster['i0'] = da_emb.i0
        da_cluster['j0'] = da_emb.j0
        da_cluster.attrs['transform_type'] = self.transform_type
        if self.transform_extra_args:
            da_cluster.attrs['transform_extra_args'] = self.transform_extra_args

        p_out = Path(self.output()['transformed_data'].fn).parent
        p_out.mkdir(exist_ok=True, parents=True)
        da_cluster.to_netcdf(self.output()['transformed_data'].fn)

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

        fn_data = src_fn.replace(
            '.nc', f".{self._build_transform_identifier()}.nc"
        )

        if self.pretrained_model:
            # we won't be resaving a pre-trained model, so there's only a
            # transformed data output, but in a subdir named after the
            # pretrained-model
            p = src_path/self.pretrained_model/fn_data
            return dict(
                transformed_data=XArrayTarget(str(p))
            )

        else:
            fn_model = self._make_transform_model_filename()
            return dict(
                model=luigi.LocalTarget(str(src_path/fn_model)),
                transformed_data=XArrayTarget(str(src_path/fn_data))
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
        if self.transform_type == "pca_hdbscan":
            pca_parent_input = yield EmbeddingTransform(
                input_path=self.input_path,
                transform_type="pca",
                pretrained_model=self.pretrained_model,
                transform_extra_args=None, # TODO: ensure enough components for HDBSCAN clustering
            )
            input_path = pca_parent_input["transformed_data"].fn
            transform_type = "hdbscan"
        else:
            input_path = self.input_path
            transform_type = self.transform_type

        parent_output = yield EmbeddingTransform(
            input_path=input_path,
            transform_type=transform_type,
            pretrained_model=self.pretrained_model,
            transform_extra_args=self.transform_extra_args,
        )

        import ipdb
        with ipdb.launch_ipdb_on_exception():
            da_emb_all = parent_output['transformed_data'].open()
            Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
            da_emb = da_emb_all.sel(scene_id=self.scene_id)
            da_emb.to_netcdf(self.output().fn)

    @property
    def input_path(self):
        return self.input().fn

    def output(self):
        model_name = Path(self.model_path).name.replace('.pkl', '')

        fn = "{}.{}_step.{}.nc".format(
            self.scene_id,
            self.step_size, self._build_transform_identifier(),
        )

        if self.pretrained_model is not None:
            name = self.pretrained_model
            p = Path(self.dataset_path)/"embeddings"/"rect"/model_name/name/fn
        else:
            p = Path(self.dataset_path)/"embeddings"/"rect"/model_name/fn

        return XArrayTarget(str(p))

    def _get_pretrained_model_path(self):
        """Return path to where pretrained transform models are expected to
        reside"""
        return Path(self.dataset_path)/"transform_models"


class CreateAllPredictionMapsDataTransformed(EmbeddingTransform):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()

    def requires(self):
        return AggregateFullDatasetImagePredictionMapData(
            dataset_path=self.dataset_path,
            model_path=self.model_path,
            step_size=self.step_size
        )

    @property
    def input_path(self):
        return self.input().fn

    def output(self):
        model_name = Path(self.model_path).name.replace('.pkl', '')
        fn = "all_embeddings.{}_step.{}.nc".format(
            self.step_size, self._build_transform_identifier(),
        )
        p_root = Path(self.dataset_path)/"embeddings"/"rect"/model_name/"components_map"
        if self.pretrained_transform_model is not None:
            p = p_root/self.pretrained_transform_model/fn
        else:
            p = p_root/fn
        return XArrayTarget(str(p))


def _annotation_plot(img, da_):
    img_kws = {}
    aug_kws = dict(add_colorbar=False, y='y')
    nx_e, ny_e = da_.shape
    img_kws['extent'] = [0, nx_e, 0, ny_e]

    if np.issubdtype(da_.dtype, np.integer):
        # use barplot if we have discrete values
        n_classes = len(np.unique(da_))
        aug_kws['colors'] = sns.color_palette(n_colors=n_classes)
        aug_kws['levels'] = n_classes-1

    ny, nx, _ = img.shape
    r = nx//ny

    sp_inches = 3
    fig, axes = plt.subplots(figsize=(r*sp_inches*2, sp_inches*2.4),
                             nrows=2, ncols=2,)

    ax = axes[1, 0]
    ax.imshow(img, **img_kws)

    ax = axes[0, 1]
    if np.issubdtype(da_.dtype, np.integer):
        # use barplot if we have discrete values
        barlist = ax.bar(*np.unique(da_, return_counts=True))
        for bar, color in zip(barlist, aug_kws['colors']):
            bar.set_color(color)
    else:
        da_.plot.hist(ax=ax)

    ax = axes[0, 0]
    ax.imshow(img, **img_kws)

    ax = axes[1, 1]
    da_.plot(ax=ax, **aug_kws)

    plt.tight_layout()

    [ax.set_aspect(1) for ax in axes.flatten()[[0, 2, 3]]]
    sns.despine()

    return fig


def _apply_transform(da, fn, transform_name):
    # stack all other dims apart from the `emb_dim`
    dims = list(da.dims)
    if "emb_dim" in dims:
        dims.remove('emb_dim')
    elif "pca_dim" in dims:
        dims.remove('pca_dim')
    else:
        raise NotImplementedError(da.dims)
    da_stacked = da.stack(dict(n=dims))
    arr = fn(X=da_stacked.T)
    if len(arr.shape) == 2:
        dims = ('n', '{}_dim'.format(transform_name))
    else:
        dims = ('n',)
    return xr.DataArray(
        arr,
        dims=dims,
        coords=dict(n=da_stacked.n)
    ).unstack('n')


def _get_img_with_extent_cropped(da_emb, img_fn):
    """
    Load the image in `img_fn`, clip the image and return the
    image extent (xy-extent if the source data coordinates are available)
    """
    img = mpimg.imread(img_fn)

    i_ = da_emb.i0.values
    # NB: j-values might be decreasing in number if we're using a
    # y-coordinate with increasing values from bottom left, so we
    # sort first
    j_ = np.sort(da_emb.j0.values)

    def get_spacing(v):
        dv_all = np.diff(v)
        assert np.all(dv_all[0] == dv_all)
        return dv_all[0]

    i_step = get_spacing(i_)
    j_step = get_spacing(j_)

    ilim = (i_.min()-i_step//2, i_.max()+i_step//2)
    jlim = (j_.min()-j_step//2, j_.max()+j_step//2)

    img = img[slice(*jlim), slice(*ilim)]

    if 'x' in da_emb.coords and 'y' in da_emb.coords:
        # get x,y and indexing (i,j) extent from data array so
        # so we can clip and plot the image correctly
        x_min = da_emb.x.min()
        y_min = da_emb.y.min()
        x_max = da_emb.x.max()
        y_max = da_emb.y.max()

        dx = get_spacing(da_emb.x.values)
        dy = get_spacing(da_emb.y.values)
        xlim = (x_min-dx//2, x_max+dx//2)
        ylim = (y_min-dy//2, y_max+dy//2)

        extent = [*xlim, *ylim]

    return img, extent


def _get_img_with_extent(da_emb, img_fn, dataset_path):
    """
    Load the image in `img_fn` and return the
    image extent (xy-extent if the source data coordinates are available)
    """
    img = mpimg.imread(img_fn)

    dataset = TripletDataset.load(dataset_path)
    domain_rect = tiler.RectTile(**dataset.extra['rectpred']['domain'])
    return img, domain_rect.get_grid_extent()


def _make_rgb(da, dims, alpha=0.5):
    def scale_zero_one(v):
        return (v-v.min())/(v.max() - v.min())
    scale = scale_zero_one

    d0, d1, d2 = da.dims
    assert d1 == 'x' and d2 == 'y'

    da_rgba = xr.DataArray(
        np.zeros((4, len(da.x), len(da.y))),
        dims=('rgba', 'x', 'y'),
        coords=dict(rgba=np.arange(4), x=da.x, y=da.y)
    )

    def _make_component(da_):
        if da_.rgba == 3:
            return alpha*np.ones_like(da_)
        else:
            return scale(da.sel({d0: dims[da_.rgba.item()]}).values)

    da_rgba = da_rgba.groupby('rgba').apply(_make_component)

    return da_rgba


class ComponentsAnnotationMapImage(luigi.Task):
    input_path = luigi.Parameter()
    components = luigi.ListParameter(default=[0, 1, 2])
    src_data_path = luigi.Parameter()
    col_wrap = luigi.IntParameter(default=2)

    def run(self):
        da_emb = xr.open_dataarray(self.input_path)

        da_emb.coords['pca_dim'] = np.arange(da_emb.pca_dim.count())

        da_emb = da_emb.assign_coords(
            x=da_emb.x/1000., y=da_emb.y/1000.,
            explained_variance=np.round(da_emb.explained_variance, 2)
        )
        da_emb.x.attrs['units'] = 'km'
        da_emb.y.attrs['units'] = 'km'

        img, img_extent = self.get_image(da_emb=da_emb)

        img_extent = np.array(img_extent)/1000.

        # find non-xy dim
        d_not_xy = next(filter(lambda d: d not in ['x', 'y'], da_emb.dims))

        N_subplots = len(self.components) + 1
        data_r = 3.
        ncols = self.col_wrap
        size = 3.

        nrows = int(np.ceil(N_subplots / ncols))
        figsize = (int(size*data_r*ncols), int(size*nrows))

        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                                 subplot_kw=dict(aspect=1), sharex=True)

        ax = axes.flatten()[0]
        ax.imshow(img, extent=img_extent)
        ax.set_title(da_emb.scene_id.item())

        for n, ax in zip(self.components, axes.flatten()[1:]):
            ax.imshow(img, extent=img_extent)
            da_ = da_emb.sel(**{d_not_xy: n})
            da_ = da_.drop(['i0', 'j0', 'scene_id'])

            da_.plot.imshow(ax=ax, y='y', alpha=0.5, add_colorbar=False)

            ax.set_xlim(*img_extent[:2])
            ax.set_ylim(*img_extent[2:])

        [ax.set_aspect(1) for ax in axes.flatten()]
        [ax.set_xlabel('') for ax in axes[:-1,:].flatten()]

        plt.tight_layout()

        fig.text(0., -0.02, "cum. explained variance: {}".format(
            np.cumsum(da_emb.explained_variance.values)
        ))

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(self.output().fn, bbox_inches='tight')
        plt.close(fig)

    def get_image(self, da_emb):
        raise NotImplementedError

    def output(self):
        image_fullpath = Path(self.input_path)
        src_path, src_fn = image_fullpath.parent, image_fullpath.name

        fn_out = src_fn.replace(
            '.nc',
            '.map.{}__comp.png'.format(
                self.src_index, "_".join([str(v) for v in self.components])
            )
        )

        p = Path(src_path)/fn_out

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
                step_size=self.step_size
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
        img_path = MakeRectRGBImage(
            dataset_path=self.dataset_path,
            scene_id=self.scene_id
        ).output().fn

        if self.crop_img:
            return _get_img_with_extent_cropped(
                da_emb, img_path
            )
        else:
            return _get_img_with_extent(
                da_emb=da_emb, img_fn=img_path,
                dataset_path=self.dataset_path
            )

    def output(self):
        model_name = Path(self.model_path).name.replace('.pkl', '')

        fn = "{}.{}_step.{}_transform.map.{}__comp.png".format(
            self.scene_id,
            self.step_size, self.transform_type,
            "_".join([str(v) for v in self.components])
        )

        p_root = Path(self.dataset_path)/"embeddings"/"rect"/model_name

        if self.pretrained_transform_model is not None:
            p = p_root/self.pretrained_transform_model/"components_map"/fn
        else:
            p = p_root/"components_map"/fn
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
            components=self.components
        )


class RGBAnnotationMapImage(luigi.Task):
    input_path = luigi.Parameter()
    rgb_components = luigi.ListParameter(default=[0, 1, 2])
    src_data_path = luigi.Parameter()
    render_tiles = luigi.BoolParameter(default=False)

    def _make_plot(self, title=None):
        da_emb = xr.open_dataarray(self.input_path)

        if len(da_emb.shape) == 3:
            # ensure non-xy dim is first
            d_not_xy = list(filter(lambda d: d not in ['x', 'y'], da_emb.dims))
            da_emb = da_emb.transpose(*d_not_xy, 'x', 'y')

        img, img_extent = self.get_image(da_emb=da_emb)

        # scale distances to km
        if da_emb.x.units == 'm' and da_emb.y.units == 'm':
            s = 1000.
            da_emb = da_emb.assign_coords(
                dict(x=da_emb.x.values/1000., y=da_emb.y.values/1000.)
            )
            da_emb.x.attrs['units'] = 'km'
            da_emb.y.attrs['units'] = 'km'

            img_extent = np.array(img_extent)/1000.
        else:
            s = 1.

        if len(da_emb.shape) == 3:
            da_rgba = _make_rgb(da=da_emb, dims=self.rgb_components, alpha=0.5)
        elif len(da_emb.shape) == 2:
            # when we have distinct classes (identified by integers) we just
            # want to map each label to a RGB color
            labels = da_emb.stack(dict(n=da_emb.dims))
            arr_rgb = skimage.color.label2rgb(label=labels.values, bg_color=(1.0, 1.0, 1.0))
            # make an RGBA array so we can apply some alpha blending later
            rgba_shape = list(arr_rgb.shape)
            rgba_shape[-1] += 1
            arr_rgba = 0.3*np.ones((rgba_shape))
            arr_rgba[...,:3] = arr_rgb
            # and put this into a DataArray, unstack to recover original dimensions
            da_rgba = xr.DataArray(
                arr_rgba,
                dims=('n', 'rgba'),
                coords=dict(n=labels.n)
            ).unstack('n')
        else:
            raise NotImplementedError(da_emb.shape)

        nrows = self.render_tiles and 4 or 3

        fig, axes = plt.subplots(figsize=(8, 3.2*nrows), nrows=nrows,
                                 subplot_kw=dict(aspect=1), sharex=True)

        ax = axes[0]
        ax.imshow(img, extent=img_extent, rasterized=True)

        ax = axes[1]
        ax.imshow(img, extent=img_extent)
        da_rgba.plot.imshow(ax=ax, rgb='rgba', y='y', rasterized=True)

        ax = axes[2]
        da_rgba[3] = 1.0
        da_rgba.plot.imshow(ax=ax, rgb='rgba', y='y', rasterized=True)

        if self.render_tiles:
            x_, y_ = xr.broadcast(da_emb.x, da_emb.y)
            axes[2].scatter(x_, y_, marker='x')

            lx = da_emb.lx_tile/s
            ly = da_emb.ly_tile/s
            ax = axes[3]
            ax.imshow(img, extent=img_extent)
            for xc, yc in zip(x_.values.flatten(), y_.values.flatten()):
                c = da_rgba.sel(x=xc, y=yc).astype(float)
                # set alpha so we can see overlapping tiles
                c[-1] = 0.2
                c_edge = np.array(c)
                c_edge[-1] = 0.5
                rect = mpatches.Rectangle(
                    (xc-lx/2., yc-ly/2), lx, ly, linewidth=1,
                    edgecolor=c_edge, facecolor=c, linestyle=':'
                )
                ax.scatter(xc, yc, color=c[:-1], marker='x')

                ax.add_patch(rect)

            def pad_lims(lim):
                l = min(lim[1] - lim[0], lim[3] - lim[2])
                return [
                    (lim[0]-0.1*l, lim[1] + l*0.1),
                    (lim[2]-0.1*l, lim[3] + l*0.1),
                ]

            xlim, ylim = pad_lims(img_extent)
        else:
            x0, y0 = da_emb.x.min(), da_emb.y.max()
            lx = da_emb.lx_tile/s
            ly = da_emb.ly_tile/s
            rect = mpatches.Rectangle(
                (x0-lx/2., y0-ly/2), lx, ly, linewidth=1,
                edgecolor='grey', facecolor='none', linestyle=':'
            )
            ax.add_patch(rect)

            xlim = img_extent[:2]
            ylim = img_extent[2:]

        [ax.set_xlim(xlim) for ax in axes]
        [ax.set_ylim(ylim) for ax in axes]
        [ax.set_aspect(1) for ax in axes]

        plt.tight_layout()

        if title is not None:
            fig.suptitle(title, y=1.05)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(self.output().fn, bbox_inches='tight')

    def run(self):
        self._make_plot()

    def get_image(self, da_emb):
        raise NotImplementedError

    def output(self):
        image_fullpath = Path(self.input_path)
        src_path, src_fn = image_fullpath.parent, image_fullpath.name

        fn_out = src_fn.replace(
            '.nc',
            '.rgb_map.{}__comp.png'.format(
                self.src_index, "_".join([str(v) for v in self.rgb_components])
            )
        )

        p = Path(src_path)/fn_out

        return luigi.LocalTarget(str(p))


class DatasetRGBAnnotationMapImage(RGBAnnotationMapImage):
    dataset_path = luigi.Parameter()
    step_size = luigi.Parameter()
    model_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    transform_type = luigi.OptionalParameter()
    transform_extra_args = luigi.OptionalParameter()
    pretrained_transform_model = luigi.OptionalParameter()
    rgb_components = luigi.ListParameter(default=[0, 1, 2])
    crop_img = luigi.BoolParameter(default=False)

    def requires(self):
        if self.transform_type is None:
            return DatasetImagePredictionMapData(
                dataset_path=self.dataset_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size
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
        model_resolution = da_emb.lx_tile/N_tile[0]/1000.
        domain_rect = dataset.extra['rectpred']['domain']
        lat0, lon0 = domain_rect['lat0'], domain_rect['lon0']

        title_parts = [
            self.scene_id,
            "(lat0, lon0)=({}, {})".format(lat0, lon0),
            "{} NN model, {} x {} tiles at {:.2f}km resolution".format(
                self.model_path.replace('.pkl', ''), N_tile[0], N_tile[1],
                model_resolution)
            ,
            "prediction RGB from {} components [{}]".format(
                da_emb.transform_type,
                ", ".join([str(v) for v in self.rgb_components]
            ))
        ]
        if self.transform_extra_args:
            title_parts.append(self.requires()._build_transform_identifier())
        title = "\n".join(title_parts)
        self._make_plot(title=title)

    @property
    def input_path(self):
        return self.input().fn

    @property
    def src_data_path(self):
        return self.dataset_path

    def get_image(self, da_emb):
        img_path = MakeRectRGBImage(
            dataset_path=self.dataset_path,
            scene_id=self.scene_id
        ).output().fn

        if self.crop_img:
            return _get_img_with_extent_cropped(
                da_emb, img_path
            )
        else:
            return _get_img_with_extent(
                da_emb=da_emb, img_fn=img_path,
                dataset_path=self.dataset_path
            )

    def output(self):
        model_name = Path(self.model_path).name.replace('.pkl', '')

        fn_parts = [
            self.scene_id,
            f"{self.step_size}_step",
            "rgb_map",
            "{}__comp".format("_".join([str(v) for v in self.rgb_components])),
        ]

        if self.transform_type:
            fn_parts.insert(2, self.requires()._build_transform_identifier())

        fn = f"{'.'.join(fn_parts)}.png"

        p_root = Path(self.dataset_path)/"embeddings"/"rect"/model_name
        if self.pretrained_transform_model is not None:
            p = p_root/self.pretrained_transform_model/fn
        else:
            p = p_root/fn
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
            rgb_components=self.rgb_components
        )
