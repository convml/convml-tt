from pathlib import Path
import platform

from PIL import Image
import numpy as np
import xarray as xr

from convml_tt.data.examples import ExampleData
import convml_tt.utils


TILE_FN_FORMAT = "{tile_id:05d}_anchor.png"


def to_pil_image(arr):
    # scale values to between zero and one
    v = (arr - arr.min()) / (arr.max() - arr.min())
    # scale to 0-254 and make uint for image
    im = Image.fromarray(np.uint8(v * 255))
    return im


def _generate_tilelist(output_path, N_tiles=10):
    output_path = Path(output_path)
    # make tiles with 5 different wavelengths, every time shifted
    # by a random amount in x and y
    output_path.mkdir(exist_ok=True)

    Nx = Ny = 256

    x_, y_ = np.linspace(0, 1, Nx), np.linspace(0, 1, Ny)
    x, y = np.meshgrid(x_, y_, indexing="ij")

    tile_id = 0
    while tile_id < N_tiles:
        frq_x = 3.14 * np.random.choice(np.arange(5))
        frq_y = 2 * frq_x
        x_offset, y_offset = np.random.normal(size=2)
        arr = np.sin((x + x_offset) * frq_x) * np.cos((y + y_offset) * frq_y)

        da = xr.DataArray(arr, dims=("x", "y"), coords=dict(x=x_, y=y_))

        img = to_pil_image(da)

        fn = TILE_FN_FORMAT.format(tile_id=tile_id)
        img.save(output_path / fn)
        tile_id += 1

    return SingleTileImageList(output_path)


def _get_simple_trained_model():
    data_path = untar_data(ExampleData.TINY10)

    monkey_patch_fastai()

    tile_path = data_path / "train"

    item_list = NPMultiImageItemList.from_folder(path=tile_path)

    src = item_list.random_split_by_pct().label_empty(embedding_length=100)

    # fix for not working multi-process training on MacOS
    # https://github.com/fastai/fastai/issues/1492
    db_kwargs = {}
    if platform.system() == "Darwin":
        db_kwargs["num_workers"] = 0

    data = (
        src.transform(
            fastai.vision.get_transforms(
                flip_vert=True,
            )
        )
        .databunch(bs=3, **db_kwargs)
        .normalize(fastai.vision.imagenet_stats)
    )

    learn = fastai.vision.create_cnn(
        data=data, base_arch=fastai.vision.models.resnet18, loss_func=loss_func
    )

    learn.fit_one_cycle(cyc_len=3, max_lr=4.0e-2)

    return learn


def _test_generation_and_interpretation():
    Nx = Ny = 256

    x_, y_ = np.linspace(0, 1, Nx), np.linspace(0, 1, Ny)
    x, y = np.meshgrid(x_, y_, indexing="ij")

    N_tiles = 50
    model = _get_simple_trained_model()

    with tempfile.TemporaryDirectory() as tmpdirname:
        tilelist = _generate_tilelist(output_path=tmpdirname)
        convml_tt.utils.get_embeddings(triplets_or_tilelist=tilelist, model=model)
