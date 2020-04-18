from pathlib import Path

import luigi
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster
import matplotlib.image as mpimg

from fastai.basic_train import load_learner
from fastai.vision import open_image

from convml_tt.architectures.triplet_trainer import monkey_patch_fastai
monkey_patch_fastai() # noqa

from convml_tt.data.sources.satellite.rectpred import MakeAllRectRGBDataArrays


def crop_fastai_im(img, i, j, nx=256, ny=256):
    img_copy = img.__class__(img._px[:,j:j+ny,i:i+nx])
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

def rect_predict(model, img, step=(10, 10), N=(256, 256)):
    """
    Produce moving-window prediction array from `img` with `model` with
    step-size defined by `step` and tile-size `N`.

    NB: j-indexing is from "top_left" i.e. is likely in the opposite order to
    what would be expected for y-axis of original image (positive being up)

    i0 and j0 coordinates denote center of each prediction tile
    """
    ny, nx = img.size
    il = FakeImagesList(None, None)
    i_ = np.arange(0, nx, step[0])
    j_ = np.arange(0, ny, step[1])

    if nx % step[0] != 0:
        i_ = i_[:-1]
    if ny % step[1] != 0:
        j_ = j_[:-1]

    for n in i_:
        for m in j_:
            il.append(crop_fastai_im(img, n, m, nx=N[0], ny=N[1]))

    result = np.stack(model.predict(il)[1]).reshape((len(i_), len(j_), -1))

    da = xr.DataArray(
        result,
        dims=('i0', 'j0', 'emb_dim'),
        coords=dict(i0=i_ + step[0]//2, j0=j_ + step[1]//2),
        attrs=dict(tile_nx=N[0], tile_ny=N[1])
    )

    return da

class CreateSingleImagePredictionMapData(luigi.Task):
    model_path = luigi.Parameter()
    image_path = luigi.Parameter()
    src_data_path = luigi.Parameter(default=None)
    step_size = luigi.IntParameter(default=10)

    def run(self):
        model_fullpath = Path(self.model_path)
        model_path, model_fn = model_fullpath.parent, model_fullpath.name
        model = load_learner(model_path, model_fn)
        img = open_image(self.image_path)

        da_pred = rect_predict(
            model=model, img=img, step=(self.step_size, self.step_size)
        )

        if self.src_data_path:
            da_src = xr.open_dataarray(self.src_data_path)
            da_pred['x'] = ('i0',), da_src.x[da_pred.i0]
            # OBS: j-indexing is positive "down" (from top-left corner) whereas
            # y-indexing is positive "up", so we have to reverse y here before
            # slicing
            da_pred['y'] = ('j0',), da_src.y[::-1][da_pred.j0]
            da_pred = da_pred.swap_dims(dict(i0='x', j0='y')).sortby(["x", "y"])

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

class CreateAllPredictionMapsData(luigi.Task):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()

    def requires(self):
        return MakeAllRectRGBDataArrays(
            dataset_path=self.dataset_path,
        )

    def run(self):
        filenames_all_scenes = self.input().read()

        prediction_tasks = []
        for (scene_da_fn, scene_image_fn) in filenames_all_scenes:
            t = CreateSingleImagePredictionMapData(
                model_path=self.model_path,
                image_path=scene_image_fn,
                src_data_path=scene_da_fn,
                step_size=self.step_size
            )
            prediction_tasks.append(t)

        prediction_outputs = yield prediction_tasks

        das = []
        for output in prediction_outputs:
            da = output.open()
            src = None
            if da.src_data_path is not None:
                src = da.src_data_path
            else:
                src = da.image_path
            da['src'] = src
            das.append(da)

        da_all = xr.concat(das, dim='src')
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
        p = Path("embeddings")/"rect"/model_name/fn
        return XArrayTarget(str(p))

class EmbeddingTransform(luigi.Task):
    input_path = luigi.Parameter()
    transform_type = luigi.Parameter()
    n_clusters = luigi.IntParameter(default=4)

    def run(self):
        da_emb = xr.open_dataarray(self.input_path)

        add_meta = None
        if self.transform_type == 'kmeans':
            fn_transform = sklearn.cluster.KMeans(
                n_clusters=self.n_clusters
            ).fit_predict
        elif self.transform_type in ['pca', 'pca_clipped']:
            model = sklearn.decomposition.PCA(n_components=self.n_clusters)
            fn_transform = model.fit_transform

            def add_meta(da):
                da['explained_variance'] = (
                    '{}_dim'.format(self.transform_type),
                    model.explained_variance_ratio_
                )
            if self.transform_type == 'pca_clipped':
                da_emb = da_emb.isel(x=slice(1, -1), y=slice(1, -1))
        else:
            raise NotImplementedError(self.transform_type)

        da_cluster = _apply_transform(
            da=da_emb, fn=fn_transform, transform_name=self.transform_type
        )

        da_cluster.attrs.update(da_emb.attrs)

        if add_meta is not None:
            add_meta(da_cluster)

        da_cluster.attrs.update(da_emb.attrs)
        da_cluster.name = 'emb'
        da_cluster['i0'] = da_emb.i0
        da_cluster['j0'] = da_emb.j0

        da_cluster.to_netcdf(self.output().fn)

    def output(self):
        src_fullpath = Path(self.input_path)
        src_path, src_fn = src_fullpath.parent, src_fullpath.name

        fn_out = src_fn.replace(
            '.nc', '.{}_transform.{}_clusters.nc'.format(
                self.transform_type, self.n_clusters
            )
        )
        return XArrayTarget(str(src_path/fn_out))

class CreateAllPredictionMapsDataTransformed(EmbeddingTransform):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()

    def requires(self):
        return CreateAllPredictionMapsData(
            dataset_path=self.dataset_path,
            model_path=self.model_path,
            step_size=self.step_size
        )

    @property
    def input_path(self):
        return self.input().fn


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

    ax = axes[1,0]
    ax.imshow(img, **img_kws)

    ax = axes[0,1]
    if np.issubdtype(da_.dtype, np.integer):
        # use barplot if we have discrete values
        barlist = ax.bar(*np.unique(da_, return_counts=True))
        for bar, color in zip(barlist, aug_kws['colors']):
            bar.set_color(color)
    else:
        da_.plot.hist(ax=ax)

    ax = axes[0,0]
    ax.imshow(img, **img_kws)

    ax = axes[1,1]
    da_.plot(ax=ax, **aug_kws)

    plt.tight_layout()

    [ax.set_aspect(1) for ax in axes.flatten()[[0,2,3]]]
    sns.despine()

    return fig

def _apply_transform(da, fn, transform_name):
    # stack all other dims apart from the `emb_dim`
    dims = list(da.dims)
    dims.remove('emb_dim')
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

class XArrayTarget(luigi.target.FileSystemTarget):
    fs = luigi.local_target.LocalFileSystem()

    def __init__(self, path, *args, **kwargs):
        super(XArrayTarget, self).__init__(path, *args, **kwargs)
        self.path = path

    def open(self, *args, **kwargs):
        # ds = xr.open_dataset(self.path, engine='h5netcdf', *args, **kwargs)
        ds = xr.open_dataset(self.path, *args, **kwargs)

        if len(ds.data_vars) == 1:
            name = list(ds.data_vars)[0]
            da = ds[name]
            da.name = name
            return da
        else:
            return ds

    @property
    def fn(self):
        return self.path

def _get_img_with_extent(da_emb, data_path):
    """
    Using the `src` attribute on the the `da` data array load up the image
    that the rectpred arrays was made from, clip the image and return the
    image extent (xy-extent if the source data coordinates are available)
    """
    src_fn = str(da_emb.src.item())

    if src_fn.endswith('.nc'):
        src_fn = src_fn.replace('.nc', '.png')

    img_fn = str(data_path/src_fn)
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

    img = img[slice(*jlim),slice(*ilim)]

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


class RGBAnnotationMapImage(luigi.Task):
    input_path = luigi.Parameter()
    src_index = luigi.IntParameter()
    rgb_components = luigi.ListParameter(default=[0,1,2])
    src_data_path = luigi.Parameter()

    def run(self):
        da_emb = xr.open_dataarray(self.input_path).isel(src=self.src_index)

        # ensure non-xy dim is first
        d_not_xy = next(filter(lambda d: d not in ['x', 'y'], da_emb.dims))
        da_emb = da_emb.transpose(d_not_xy, 'x', 'y')

        import ipdb
        with ipdb.launch_ipdb_on_exception():
            da_rgba = _make_rgb(da=da_emb, dims=self.rgb_components, alpha=0.5)

        fig, axes = plt.subplots(figsize=(10, 4*3), nrows=3,
                                 subplot_kw=dict(aspect=1), sharex=True)

        img, img_extent = _get_img_with_extent(
            da_emb, Path(self.src_data_path)
        )

        ax = axes[0]
        ax.imshow(img, extent=img_extent)

        ax = axes[1]
        ax.imshow(img, extent=img_extent)
        da_rgba.plot.imshow(ax=ax, rgb='rgba', y='y')

        ax = axes[2]
        da_rgba[3] = 1.0
        da_rgba.plot.imshow(ax=ax, rgb='rgba', y='y')

        plt.savefig(self.output().fn)

    def output(self):
        image_fullpath = Path(self.input_path)
        src_path, src_fn = image_fullpath.parent, image_fullpath.name

        fn_out = src_fn.replace(
            '.nc',
            '.src_{}.rgb_map.{}__comp.png'.format(
                self.src_index, "_".join([str(v) for v in self.rgb_components])
            )
        )

        p = Path(src_path)/fn_out

        return luigi.LocalTarget(str(p))


class DatasetRGBAnnotationMapImage(RGBAnnotationMapImage):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()
    transform_type = luigi.Parameter(default=None)
    rgb_components = luigi.Parameter(default=[0,1,2])

    src_index = luigi.IntParameter()

    def requires(self):
        if self.transform_type is None:
            return CreateAllPredictionMapsData(
                dataset_path=self.dataset_path,
                model_path=self.model_path,
                step_size=self.step_size
            )
        else:
            return CreateAllPredictionMapsDataTransformed(
                dataset_path=self.dataset_path,
                model_path=self.model_path,
                step_size=self.step_size,
                transform_type=self.transform_type,
                n_clusters=max(self.rgb_components)+1,
            )

    @property
    def input_path(self):
        return self.input().fn

    @property
    def src_data_path(self):
        return self.dataset_path


class AllDatasetRGBAnnotationMapImages(luigi.Task):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()
    transform_type = luigi.Parameter(default=None)
    rgb_components = luigi.Parameter(default=[0,1,2])

    def requires(self):
        if self.transform_type is None:
            return CreateAllPredictionMapsData(
                dataset_path=self.dataset_path,
                model_path=self.model_path,
                step_size=self.step_size
            )
        else:
            return CreateAllPredictionMapsDataTransformed(
                dataset_path=self.dataset_path,
                model_path=self.model_path,
                step_size=self.step_size,
                transform_type=self.transform_type,
                n_clusters=max(self.rgb_components)+1,
            )

    def _get_runtime_tasks(self):
        da_emb = self.input().open()
        return [
            RGBAnnotationMapImage(
                input_path=self.input().fn,
                src_index=i,
                rgb_components=self.rgb_components,
                src_data_path=self.dataset_path
            )
            for i in range(int(da_emb.src.count()))
        ]

    def run(self):
        yield self._get_runtime_tasks()

    def output(self):
        if not self.input().exists():
            return luigi.LocalTarget('__fake__file__.nc')
        else:
            return [t.output() for t in self._get_runtime_tasks()]
