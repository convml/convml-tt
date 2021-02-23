import numpy as np
from torch.utils.data import DataLoader

from convml_tt.data.dataset import TileType
from convml_tt.data.examples import ExampleData, fetch_example_dataset
from convml_tt.system import (
    Tile2Vec,
    get_single_tile_dataset,
)
from convml_tt.utils import get_embeddings
from convml_tt.interpretation import plots as interpretation_plot


def test_get_embeddings():
    model = Tile2Vec(pretrained=True)

    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    dataset = get_single_tile_dataset(data_dir=data_path, stage="train", tile_type=TileType.ANCHOR)

    # direct
    dl_predict = DataLoader(dataset, batch_size=32)
    batched_results = [model.forward(x_batch) for x_batch in dl_predict]
    results = np.vstack([v.cpu().detach().numpy() for v in batched_results])

    # via utily function
    da_embeddings = get_embeddings(tile_dataset=dataset, model=model, prediction_batch_size=32)

    Ntiles, Ndim = results.shape

    assert int(da_embeddings.tile_id.count()) == Ntiles
    assert int(da_embeddings.emb_dim.count()) == Ndim


def test_overview_plot():
    data_path = fetch_example_dataset(dataset=ExampleData.SMALL100)
    tile_dataset = get_single_tile_dataset(data_dir=data_path, stage="train", tile_type=TileType.ANCHOR)
    interpretation_plot.grid_overview(tile_dataset=tile_dataset, points=10)
