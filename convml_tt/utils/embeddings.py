"""
Utility functions for the triplet-trainer
"""
import multiprocessing
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader

from ..data.dataset import (
    ImageSingletDataset,
    ImageTripletDataset,
    MovingWindowImageTilingDataset,
    TileType,
)


def get_embeddings(
    tile_dataset: [ImageSingletDataset, ImageTripletDataset],
    model,
    prediction_batch_size=32,
    n_worker_cpu_cores="all",
):
    """
    Use the provided model to calculate the embeddings for all tiles of a
    specific `tile_type` in the given `data_dir`. If you run out of memory
    reduce the `prediction_batch_size` (you may also increase it to generate
    predictions faster while using more RAM).

    If a GPU is available it will be used. For now we only use a single GPU
    even if multiple are available. By default we will use all available cpu
    cores for the dataloader.

    For a moving window dataset (data.datasets.MovingWindowImageTilingDataset)
    the data will be reshaped to be 2D as the input image. The i0 and j0
    coordinates denote center of each prediction tile. NB: j-indexing is from
    "top_left" i.e. is likely in the opposite order to what would be expected
    for y-axis of original image (positive being up)

    """
    if len(tile_dataset) == 0:
        raise Exception("No tiles in the provided dataset")

    if isinstance(tile_dataset, ImageTripletDataset):
        das = []
        for tile_type in TileType:
            tile_dataset_tiletype = tile_dataset.make_singlet_dataset(
                tile_type=tile_type
            )
            da_embs_tiletype = get_embeddings(
                tile_dataset=tile_dataset_tiletype,
                model=model,
                prediction_batch_size=prediction_batch_size,
                n_worker_cpu_cores=n_worker_cpu_cores,
            )
            da_embs_tiletype["tile_type"] = tile_type.name.lower()
            das.append(da_embs_tiletype)

        da_embeddings = xr.concat(das, dim="tile_type")
        return da_embeddings

    if n_worker_cpu_cores == "all":
        n_worker_cpu_cores = multiprocessing.cpu_count()

    tile_dataloader = DataLoader(
        dataset=tile_dataset,
        batch_size=prediction_batch_size,
        num_workers=n_worker_cpu_cores,
    )
    batched_results = []

    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = 0
    trainer = pl.Trainer(gpus=gpus)

    # by using the index of the source pd.DataFrame here we can ensure that the
    # tile_id comes from the filename rather than simple the number for each
    # sample in the dataset
    tile_ids = tile_dataset.index
    coords = dict(tile_id=tile_ids)

    if isinstance(tile_dataset, ImageSingletDataset):
        dims = ("tile_id", "emb_dim")
    elif isinstance(tile_dataset, ImageTripletDataset):
        dims = ("tile_type", "tile_id", "emb_dim")
        coords["tile_type"] = ["anchor", "neighbor", "distant"]
    else:
        raise NotImplementedError(type(tile_dataset))

    batched_results = []
    # XXX: this is a hack. There appears to be somesort of race condition
    # inside of pytorch-lightning that means that sometimes calling
    # Trainer.predict returns an empty list. So, we'll just run it again until
    # we get something back
    while len(batched_results) == 0:
        batched_results = trainer.predict(model=model, dataloaders=tile_dataloader)

    embeddings = np.vstack([batch.detach().numpy() for batch in batched_results])

    attrs = {}
    if getattr(tile_dataset, "tile_type", None):
        attrs["tile_type"] = tile_dataset.tile_type.name
    if getattr(tile_dataset, "stage", None):
        attrs["stage"] = tile_dataset.stage
    if getattr(tile_dataset, "data_dir", None):
        attrs["data_dir"] = str(Path(tile_dataset.data_dir).absolute())

    da_emb = xr.DataArray(embeddings, dims=dims, coords=coords, attrs=attrs)

    if isinstance(tile_dataset, MovingWindowImageTilingDataset):
        da_emb = tile_dataset.add_tiling_coords_to_embedding_dataarray(da_emb)

    return da_emb


def make_sliding_tile_model_predictions(
    model,
    tile_dataset: MovingWindowImageTilingDataset,
    prediction_batch_size=32,
):
    raise NotImplementedError(
        "`make_sliding_tile_model_predictions` has been removed and its functionality"
        " added to `utils.get_embeddings`. Please change to use `utils.get_embeddings`"
        " instead."
    )
