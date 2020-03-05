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

    return result

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

        pred = rect_predict(model=model, img=img,
            step=(self.step_size, self.step_size)
        )

        import ipdb
        with ipdb.launch_ipdb_on_exception():
            kws = {}
            if self.src_data_path is not None:
                da_source = xr.open_dataarray(self.src_data_path)
                s = slice(self.step_size//2, None, self.step_size)
                nx_p, ny_p, _ = pred.shape
                x_ = da_source.x[s][:nx_p]
                y_ = da_source.y[s][:ny_p]
                kws['coords'] = dict(x=x_, y=y_)

            da_pred = xr.DataArray(
                pred, dims=('x', 'y', 'emb_dim'), **kws
            )
        da_pred.to_netcdf(self.output().fn)

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        fn_out = image_fn.replace('.png', '.embeddings.{}_step.nc'.format(self.step_size))
        return luigi.LocalTarget(fn_out)

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

def _apply_transform(da, fn):
    da_stacked = da.stack(dict(n=('x','y')))
    arr = fn(X=da_stacked.T)
    if len(arr.shape) == 2:
        dims = ('n', 'tsne_dim')
    else:
        dims = ('n',)
    return xr.DataArray(
        arr,
        dims=dims,
        coords=dict(n=da_stacked.n)
    ).unstack('n')

class AnnotationMapImage(luigi.Task):
    model_path = luigi.Parameter()
    image_path = luigi.Parameter()
    src_data_path = luigi.Parameter(default=None)
    step_size = luigi.IntParameter(default=10)
    n_clusters = luigi.IntParameter(default=4)

    def requires(self):
        return CreateSingleImagePredictionMapData(
            model_path=self.model_path,
            image_path=self.image_path,
            src_data_path=self.src_data_path,
            step_size=self.step_size
        )

    def run(self):
        img = mpimg.imread(self.image_path)
        da_emb = xr.open_dataarray(self.input().fn)
        fn_transform = sklearn.cluster.KMeans(n_clusters=self.n_clusters).fit_predict
        da_cluster = _apply_transform(da=da_emb, fn=fn_transform)
        fig = _annotation_plot(img, da_cluster)
        plt.savefig(self.output().fn)

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        fn_out = image_fn.replace('.png', 
            '.embedding_map.{}_step.{}_clusters.png'.format(
                self.step_size, self.n_clusters
                )
            )
        return luigi.LocalTarget(fn_out)

class CreatePredictionMap(luigi.Task):
    source_data_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    scene_num = luigi.IntParameter()
    dx = luigi.FloatParameter(default=200.0e3/256)
