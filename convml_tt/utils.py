import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from PIL import Image
from tqdm import tqdm

from .data.dataset import ImageSingletDataset, TileType


def get_embeddings(tile_dataloader, model):
    """
    Use the provided model to calculate the embeddings for all tiles of a
    specific `tile_type` in the given `data_dir`.
    """
    batched_results = [model.forward(x_batch) for x_batch in tile_dataloader]
    embeddings = np.vstack([v.cpu().detach().numpy() for v in batched_results])

    tile_ids = np.arange(len(tile_dataloader.dataset))

    dims = ("tile_id", "emb_dim")
    coords = dict(tile_id=tile_ids)
    attrs = dict(source_path=str(tile_dataloader.dataset.full_path.absolute()))

    return xr.DataArray(embeddings, dims=dims, coords=coords, attrs=attrs)


class ImageLoader:
    def __init__(self, path):
        self.path = Path(path)

    def __getitem__(self, n):
        img_path = self.path / "{:05d}_anchor.png".format(n)
        return Image.open(img_path)


def get_triplets_from_embeddings(embeddings):
    return ImageLoader(embeddings.source_path)
