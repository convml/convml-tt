import xarray as xr
import numpy as np
import torch
from tqdm import tqdm

import warnings

from pathlib import Path

from PIL import Image

from .architectures.triplet_trainer import TileType

def get_embeddings(triplets, model, tile_type=TileType.ANCHOR):
    """
    Use the provided model to calculate the embeddings for the given set of
    input triplets. If `tile_type` is None the embedding for all three tiles of
    the triplet will be returned.
    """
    def _get_embedding(image_set, *args, **kwargs):
        try:
            v = model.predict(image_set[tile_type])[1]
        except RuntimeError:
            # "... eturns a tuple of three things: the object predicted (with
            # the class in this instance), the underlying data (here the
            # corresponding index) and the raw probabilities"
            # https://docs.fast.ai/tutorial.inference.html#Vision
            # we only want the underlying data (the three embeddings for the
            # triplet provided)
            v = model.predict(image_set)[1]
        return image_set.id, v

    triplet_ids_and_embeddings = [
        _get_embedding(image_set) for image_set in tqdm(triplets)
    ]

    triplet_ids, embeddings = zip(*triplet_ids_and_embeddings)

    coords=dict(tile_id=tile_id, emb_dim=np.arange(embeddings.shape[1]))
    if tile_type is not None:
        embeddings = embeddings[tile_type]
        dims = ('tile_id', 'emb_dim')
    else:
        dims = ('tile_id', 'emb_dim', 'tile_type')
        coords['tile_type'] = TileType.NAMES

    embeddings = np.asarray(torch.stack(embeddings).squeeze())
    # we're picking out the anchor tile above
    tile_id = np.asarray(triplet_ids).astype(int)

    return xr.DataArray(
        embeddings, dims=dims, coords=coords,
        attrs=dict(tile_used=TileType.NAMES[tile_type],
        source_path=str(triplets.path.absolute())
        )
    )

def get_encodings(triplets, model, tile_type=TileType.ANCHOR):
    warnings.warn("`get_encodings` has been renamed `get_embeddings`")
    return get_embeddings(triplets=tripelts, model=model, tile_type=tile_type)


class ImageLoader:
    def __init__(self, path):
        self.path = Path(path)

    def __getitem__(self, n):
        img_path = self.path/"{:05d}_anchor.png".format(n)
        return Image.open(img_path)

def get_triplets_from_encodings(encodings):
    return ImageLoader(encodings.source_path)
