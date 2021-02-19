import warnings
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from PIL import Image
from tqdm import tqdm

from .data.dataset import TileType


def get_embeddings(triplets_or_tilelist, model, tile_type=TileType.ANCHOR):
    """
    Use the provided model to calculate the embeddings for the given set of
    input triplets or tilelist. If `tile_type` is None the embedding for all
    three tiles of the triplet will be returned.
    """
    raise NotImplementedError
    if isinstance(triplets_or_tilelist, SingleTileImageList):
        il = triplets_or_tilelist
        embeddings = np.stack([v.cpu() for v in model.predict(il)[1]])
        dims = ('tile_id', 'emb_dim')
        coords = dict(tile_id=il.tile_ids)
        attrs = dict(source_path=str(il.src_path.absolute()))
    else:
        triplets = triplets_or_tilelist
        def _get_embedding(image_set, *args, **kwargs):
            # "`predict` returns a tuple of three things: the object predicted (with
            # the class in this instance), the underlying data (here the
            # corresponding index) and the raw probabilities"
            # https://docs.fast.ai/tutorial.inference.html#Vision
            # we only want the underlying data (the three embeddings for the
            # triplet provided)
            T_emb = model.predict(image_set)[1]
            v = np.array([np.array(a.cpu()) for a in T_emb])

            return image_set.id, v

        triplet_ids_and_embeddings = [
            _get_embedding(image_set) for image_set in tqdm(triplets)
        ]

        triplet_ids, embeddings = zip(*triplet_ids_and_embeddings)
        embeddings = np.asarray(np.stack(embeddings).squeeze())

        tile_id = np.asarray(triplet_ids).astype(int)
        coords=dict(tile_id=tile_id, emb_dim=np.arange(embeddings.shape[-1]))
        attrs=dict(source_path=str(triplets.path.absolute()))
        if tile_type is not None:
            embeddings = embeddings[:,tile_type.value]
            dims = ('tile_id', 'emb_dim')
            attrs['tile_used'] = tile_type.name.lower()
        else:
            dims = ('tile_id', 'tile_type', 'emb_dim')
            coords['tile_type'] = [s.lower() for s in TileType.__members__.keys()]

    return xr.DataArray(
        embeddings, dims=dims, coords=coords, attrs=attrs
    )

def get_encodings(triplets, model, tile_type=TileType.ANCHOR):
    warnings.warn("`get_encodings` has been renamed `get_embeddings`")
    return get_embeddings(triplets=triplets, model=model, tile_type=tile_type)


class ImageLoader:
    def __init__(self, path):
        self.path = Path(path)

    def __getitem__(self, n):
        img_path = self.path/"{:05d}_anchor.png".format(n)
        return Image.open(img_path)

def get_triplets_from_embeddings(embeddings):
    return ImageLoader(embeddings.source_path)
