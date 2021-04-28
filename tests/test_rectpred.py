from pathlib import Path

from PIL import Image
import xarray as xr
import numpy as np

from convml_tt.interpretation.rectpred.data import make_sliding_tile_model_predictions
from convml_tt.system import TripletTrainerModel
from convml_tt.interpretation.rectpred.plot import make_rgb
from convml_tt.interpretation.rectpred.transform import apply_transform

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
    da_emb_rect = make_sliding_tile_model_predictions(img=img, model=model, step=step)

    nx_img, ny_img = img.size

    # number of tiles expected in each direction
    nxt = (nx_img - N_tile[0] + step[0]) // step[0]
    nyt = (ny_img - N_tile[1] + step[1]) // step[1]

    assert da_emb_rect.emb_dim.count() == model.n_embedding_dims
    assert da_emb_rect.i0.count() == nxt
    assert da_emb_rect.j0.count() == nyt


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
