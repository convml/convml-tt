from pathlib import Path

import luigi
import numpy as np
import xarray as xr

from fastai.basic_train import load_learner
from fastai.vision import open_image

from tqdm import tqdm

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

    for n in tqdm(i_):
        for m in tqdm(j_):
            il.append(crop_fastai_im(img, n, m))

    result = np.stack(model.predict(il)[1]).reshape((len(i_), len(j_), -1))

    return result

class CreateSingleImagePredictionMap(luigi.Task):
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

        fn_out = image_fn.replace('.png', '.embeddings.nc')
        return luigi.LocalTarget(fn_out)


class CreatePredictionMap(luigi.Task):
    source_data_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    scene_num = luigi.IntParameter()
    dx = luigi.FloatParameter(default=200.0e3/256)
