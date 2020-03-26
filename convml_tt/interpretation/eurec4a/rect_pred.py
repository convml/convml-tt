from pathlib import Path

import luigi
import numpy as np
import xarray as xr
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster

from fastai.basic_train import load_learner
from fastai.vision import open_image

from convml_tt.architectures.triplet_trainer import monkey_patch_fastai
monkey_patch_fastai()

from convml_tt.data.sources.satellite.rectpred import MakeAllRectRGBDataArrays



def crop_fastai_im(img, i, j, nx=256, ny=256):
    img_copy = img.__class__(img._px[:,j:j+ny,i:i+nx])
    # From PIL docs: The crop rectangle, as a (left, upper, right, lower)-tuple.
    #print(i, 0, nx+i, nx)
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

def rect_predict(model, img, step=(10, 10)):
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
            il.append(crop_fastai_im(img, n, m))

    result = np.stack(model.predict(il)[1]).reshape((len(i_), len(j_), -1))

    # for some reason it appears that the y-coordinate gets reversed...
    result = result[:,::-1]

    da = xr.DataArray(
        result,
        dims=('i0', 'j0', 'emb_dim'),
        coords=dict(i0=i_, j0=j_),
        attrs=dict(xstep=step[0], ystep=step[1])
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

        da_pred = rect_predict(model=model, img=img,
            step=(self.step_size, self.step_size)
        )

        if self.src_data_path:
            da_src = xr.open_dataarray(self.src_data_path)

            da_pred['x'] = da_src.x[da_pred.i0 + da_pred.xstep//2]
            da_pred['y'] = da_src.y[da_pred.j0 + da_pred.ystep//2]

        da_pred = da_pred.swap_dims(dict(i0='x', j0='y'))

        da_pred.attrs['model_path'] = self.model_path
        da_pred.attrs['image_path'] = self.image_path
        da_pred.attrs['src_data_path'] = self.src_data_path

        da_pred.to_netcdf(self.output().fn)

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        fn_out = image_fn.replace('.png', '.embeddings.{}_step.nc'.format(self.step_size))
        return XArrayTarget(str(image_path/fn_out))

class CreateAllPredictionMapsData(luigi.Task):
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()

    def _get_dataset(self):
        return FixedTimeRangeSatelliteTripletDataset.load(self.dataset_path)

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
                src = da.iamge_path
            da['src'] = src
            das.append(da)

        da_all = xr.concat(das, dim='src')
        da_all.name = 'emb'
        da_all.attrs['step_size'] = self.step_size
        da_all.attrs['model_path'] = self.model_path

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
            fn_transform = sklearn.cluster.KMeans(n_clusters=self.n_clusters).fit_predict
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

        da_cluster = _apply_transform(da=da_emb, fn=fn_transform,
            transform_name=self.transform_type
        )

        if add_meta is not None:
            add_meta(da_cluster)

        da_cluster.to_netcdf(self.output().fn)

    def output(self):
        src_fullpath = Path(self.input_path)
        src_path, src_fn = src_fullpath.parent, src_fullpath.name

        fn_out = src_fn.replace('.nc',
            '.{}_transform.{}_clusters.nc'.format(
                self.transform_type, self.n_clusters
            )
        )
        return XArrayTarget(str(src_path/fn_out))

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
    fig, axes = plt.subplots(figsize=(r*sp_inches*2, sp_inches*2.4), nrows=2, ncols=2, )

    ax = axes[1,0]
    ax.imshow(img, **img_kws)
    c_pl = da_.plot(ax=ax, alpha=0.5, **aug_kws)

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

class TransformedMapPrediction(luigi.Task):
    model_path = luigi.Parameter()
    image_path = luigi.Parameter()
    src_data_path = luigi.Parameter(default=None)
    step_size = luigi.IntParameter(default=10)
    n_clusters = luigi.IntParameter(default=4)
    transform_type = luigi.Parameter(default='tsne')

    def requires(self):
        assert self.transform_type in ["kmeans", "pca"]
        return CreateSingleImagePredictionMapData(
            model_path=self.model_path,
            image_path=self.image_path,
            src_data_path=self.src_data_path,
            step_size=self.step_size
        )

    def run(self):
        img = mpimg.imread(self.image_path)
        da_emb = xr.open_dataarray(self.input().fn)
        if self.transform_type == 'kmeans':
            fn_transform = sklearn.cluster.KMeans(n_clusters=self.n_clusters).fit_predict
        elif self.transform_type == 'pca':
            fn_transform = sklearn.decomposition.PCA(n_components=self.n_clusters).fit_transform
        else:
            raise NotImplementedError(self.transform_type)
        da_cluster = _apply_transform(da=da_emb, fn=fn_transform,
            transform_name=self.transform_type
        )

        da_cluster.to_netcdf(self.output().fn)

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        fn_out = image_fn.replace('.png', 
            '.embedding_map.{}_step.{}_transform.{}_clusters.nc'.format(
                self.step_size, self.transform_type, self.n_clusters
                )
            )
        return XArrayTarget(str(image_path/fn_out))

class AnnotationMapImage(luigi.Task):
    model_path = luigi.Parameter()
    image_path = luigi.Parameter()
    src_data_path = luigi.Parameter(default=None)
    step_size = luigi.IntParameter(default=10)
    n_clusters = luigi.IntParameter(default=4)
    transform_type = luigi.Parameter(default='tsne')

    def requires(self):
        return TransformedMapPrediction(
            model_path=self.model_path,
            image_path=self.image_path,
            src_data_path=self.src_data_path,
            step_size=self.step_size,
            n_clusters=self.n_clusters,
            transform_type=self.transform_type
        )

    def run(self):
        da_cluster = self.input().open()
        fig = _annotation_plot(img, da_cluster)
        plt.savefig(self.output().fn)

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        fn_out = image_fn.replace('.png', 
            '.embedding_map.{}_step.{}_transform.{}_clusters.png'.format(
                self.step_size, self.transform_type, self.n_clusters
                )
            )
        return luigi.LocalTarget(fn_out)

class CreatePredictionMap(luigi.Task):
    source_data_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    scene_num = luigi.IntParameter()
    dx = luigi.FloatParameter(default=200.0e3/256)
