import tempfile
from pathlib import Path

import luigi
import numpy as np
import xarray as xr
from PIL import Image

from convml_tt.data.dataset import MovingWindowImageTilingDataset
from convml_tt.data.examples import PretrainedModel, fetch_pretrained_model
from convml_tt.data.sources.examples import ExampleDatasource, fetch_example_datasource
from convml_tt.data.transforms import get_transforms as get_model_transforms
from convml_tt.interpretation.plots import annotated_scatter_plot
from convml_tt.interpretation.rectpred.pipeline.data import (
    AggregateFullDatasetImagePredictionMapData,
)
from convml_tt.interpretation.rectpred.plot import make_rgb
from convml_tt.interpretation.rectpred.transform import apply_transform
from convml_tt.system import TripletTrainerModel
from convml_tt.utils import get_embeddings

DOC_PATH = Path(__file__).parent.parent / "doc"
RECTPRED_IMG_EXAMPLE_PATH = DOC_PATH / "goes16_202002051400.png"

if not RECTPRED_IMG_EXAMPLE_PATH.exists():
    raise Exception("Can't find example image to use for rectpred tests")


def test_rectpred_sliding_window_inference():
    # use a model with default resnet weights to generate some embedding
    # vectors to plot with
    backbone_arch = "resnet18"
    model = TripletTrainerModel(pretrained=True, base_arch=backbone_arch)
    # TODO: make this a property of the model
    N_tile = (256, 256)

    img = Image.open(RECTPRED_IMG_EXAMPLE_PATH)
    step = (500, 200)
    transforms = get_model_transforms(
        step="predict", normalize_for_arch=model.base_arch
    )
    tile_dataset = MovingWindowImageTilingDataset(
        img=img, transform=transforms, step=step, N_tile=N_tile
    )
    da_emb_rect = get_embeddings(
        tile_dataset=tile_dataset, model=model
    )

    nx_img, ny_img = img.size

    # number of tiles expected in each direction
    nxt = (nx_img - N_tile[0] + step[0]) // step[0]
    nyt = (ny_img - N_tile[1] + step[1]) // step[1]

    assert da_emb_rect.emb_dim.count() == model.n_embedding_dims
    assert da_emb_rect.i0.count() == nxt
    assert da_emb_rect.j0.count() == nyt

    # try creating an annotated scatter with the embedding
    da_x = da_emb_rect.sel(emb_dim=0)
    da_y = da_emb_rect.sel(emb_dim=0)
    annotated_scatter_plot(
        x=da_x, y=da_y, points=5, tile_dataset=tile_dataset, autopos_method=None
    )


def test_apply_transform_rect():
    n_dim = 20
    nx, ny = 10, 8
    arr = np.random.random((n_dim, nx, ny))
    da_emb = xr.DataArray(
        arr,
        coords=dict(emb_dim=np.arange(n_dim), i=np.arange(nx), j=np.arange(ny)),
        dims=("emb_dim", "i", "j"),
    )

    apply_transform(da=da_emb, transform_type="pca")


def test_make_rgb_rect():
    n_dim = 20
    nx, ny = 10, 8
    arr = np.random.random((n_dim, nx, ny))
    da_emb = xr.DataArray(
        arr,
        coords=dict(emb_dim=np.arange(n_dim), i=np.arange(nx), j=np.arange(ny)),
        dims=("emb_dim", "i", "j"),
    )

    make_rgb(da=da_emb, emb_dim=[1, 3, 2])


def test_make_rect_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        datasource_path = fetch_example_datasource(
            ExampleDatasource.EUREC4A_SMALL, data_dir=tmpdir
        )
        model_path = fetch_pretrained_model(PretrainedModel.FIXED_NORM_STAGE2)
        task_rect_data = AggregateFullDatasetImagePredictionMapData(
            data_path=datasource_path,
            model_path=model_path,
            step_size=100,
            generate_tile_images=True,
        )
        luigi.build([task_rect_data], local_scheduler=True)
