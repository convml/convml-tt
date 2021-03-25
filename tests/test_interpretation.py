import numpy as np
from torch.utils.data import DataLoader

from convml_tt.data.dataset import TileType, ImageSingletDataset
from convml_tt.data.examples import ExampleData, fetch_example_dataset
from convml_tt.data.transforms import get_transforms
from convml_tt.interpretation import plots as interpretation_plot
from convml_tt.system import TripletTrainerModel
from convml_tt.utils import get_embeddings


def test_get_embeddings():
    backbone_arch = "resnet18"
    model = TripletTrainerModel(pretrained=True, base_arch=backbone_arch)

    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    dataset = ImageSingletDataset(
        data_dir=data_path,
        stage="train",
        tile_type=TileType.ANCHOR,
        transform=get_transforms(step="predict", normalize_for_arch=backbone_arch),
    )

    # direct
    dl_predict = DataLoader(dataset, batch_size=32)
    batched_results = [model.forward(x_batch) for x_batch in dl_predict]
    results = np.vstack([v.cpu().detach().numpy() for v in batched_results])

    # via utily function
    da_embeddings = get_embeddings(
        tile_dataset=dataset, model=model, prediction_batch_size=16
    )

    Ntiles, Ndim = results.shape

    assert int(da_embeddings.tile_id.count()) == Ntiles
    assert int(da_embeddings.emb_dim.count()) == Ndim


def test_grid_overview_plot():
    data_path = fetch_example_dataset(dataset=ExampleData.SMALL100)
    tile_dataset = ImageSingletDataset(
        data_dir=data_path, stage="train", tile_type=TileType.ANCHOR
    )
    interpretation_plot.grid_overview(tile_dataset=tile_dataset, points=10)


def test_dendrogram_plot():
    # use a model with default resnet weights to generate some embedding
    # vectors to plot with
    backbone_arch = "resnet18"
    model = TripletTrainerModel(pretrained=True, base_arch=backbone_arch)

    data_path = fetch_example_dataset(dataset=ExampleData.SMALL100)
    tile_dataset = ImageSingletDataset(
        data_dir=data_path,
        stage="train",
        tile_type=TileType.ANCHOR,
        transform=get_transforms(step="predict", normalize_for_arch=backbone_arch),
    )

    da_embeddings = get_embeddings(
        tile_dataset=tile_dataset, model=model, prediction_batch_size=16
    )
    interpretation_plot.dendrogram(da_embeddings=da_embeddings)


def test_annotated_scatter_plot():
    # use a model with default resnet weights to generate some embedding
    # vectors to plot with
    backbone_arch = "resnet18"
    model = TripletTrainerModel(pretrained=True, base_arch=backbone_arch)

    data_path = fetch_example_dataset(dataset=ExampleData.SMALL100)
    tile_dataset = ImageSingletDataset(
        data_dir=data_path,
        stage="train",
        tile_type=TileType.ANCHOR,
        transform=get_transforms(step="predict", normalize_for_arch=backbone_arch),
    )

    da_embeddings = get_embeddings(
        tile_dataset=tile_dataset, model=model, prediction_batch_size=16
    )
    x = da_embeddings.isel(emb_dim=0)
    y = da_embeddings.isel(emb_dim=1)
    interpretation_plot.annotated_scatter_plot(x=x, y=y, points=10)
