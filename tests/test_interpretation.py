import numpy as np

from convml_tt.data.dataset import TileType
from convml_tt.data.examples import ExampleData, get_example_dataset
from convml_tt.system import (
    Tile2Vec,
    get_single_tile_dataloader,
)
from convml_tt.utils import get_embeddings


def test_get_embeddings():
    model = Tile2Vec(pretrained=True)

    data_path = get_example_dataset(dataset=ExampleData.TINY10)
    dl_predict = get_single_tile_dataloader(data_dir=data_path, stage="train", tile_type=TileType.ANCHOR)

    # direct
    batched_results = [model.forward(x_batch) for x_batch in dl_predict]
    results = np.vstack([v.cpu().detach().numpy() for v in batched_results])

    # via utily function
    da_embeddings = get_embeddings(tile_dataloader=dl_predict, model=model)

    Ntiles, Ndim = results.shape

    assert int(da_embeddings.tile_id.count()) == Ntiles
    assert int(da_embeddings.emb_dim.count()) == Ndim


# TODO: reanable this test after refactoring satellite specific code
# def _test_tile_loading():
#     data_path = untar_data(ExampleData.TINY10)
#
#     monkey_patch_fastai()
#
#     tile_path = data_path / "train"
#     triplets = NPMultiImageItemList.from_folder(path=tile_path)
#
#     for tile_type in TileType:
#         convml_tt.data.triplets.load_tile_definitions(
#             triplets=triplets, tile_type=tile_type
#         )
