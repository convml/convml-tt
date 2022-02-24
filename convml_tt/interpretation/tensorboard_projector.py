# from torch.utils.tensorboard import SummaryWriter
import hashlib
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tensorboardX import SummaryWriter
from tqdm import tqdm

from convml_tt.data.dataset import ImageTripletDataset
from convml_tt.data.examples import (
    ExampleData,
    PretrainedModel,
    fetch_example_dataset,
    load_pretrained_model,
)
from convml_tt.data.transforms import get_transforms
from convml_tt.system import TripletTrainerModel
from convml_tt.utils import get_embeddings


def vector_norm(x, dim, ord=None):
    return xr.apply_ufunc(
        np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1}
    )


def _make_hash(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def main(model_path, data_path):
    if model_path.startswith("ex://"):
        model_name = model_path[5:]
        available_models = [m.name for m in list(PretrainedModel)]
        if model_name not in available_models:
            raise Exception(
                f"pretrained model `{model_name}` not found."
                f" available models: {', '.join(available_models)}"
            )
        model = load_pretrained_model(PretrainedModel[model_name])
    else:
        model = TripletTrainerModel.load_from_checkpoint(model_path)

    if data_path.startswith("ex://"):
        dset_name = data_path[5:]
        available_dsets = [m.name for m in list(ExampleData)]
        if model_name not in available_models:
            raise Exception(
                f"example dataset `{dset_name}` not found."
                f" available dataset: {', '.join(available_dsets)}"
            )
        model = load_pretrained_model(PretrainedModel[model_name])
        data_path = fetch_example_dataset(dataset=ExampleData[dset_name])
    transforms = get_transforms(step="predict", normalize_for_arch=model.base_arch)
    dset = ImageTripletDataset(data_dir=data_path, stage="train", transform=transforms)

    embs_id = _make_hash(f"{model_path}__{data_path}")
    fpath_embs = Path(f"embs-{embs_id}.nc")

    if not fpath_embs.exists():
        da_embs = get_embeddings(
            tile_dataset=dset, model=model, prediction_batch_size=64
        )
        da_embs.to_netcdf(fpath_embs)
    else:
        da_embs = xr.open_dataarray(fpath_embs)
        print(f"using saved embeddings from `{fpath_embs}`")

    _save_embeddings(da_embs=da_embs, dset=dset)


def main2(embs_path):
    da_embs = xr.open_dataarray(embs_path)
    data_path = da_embs.data_dir
    dset = ImageTripletDataset(data_dir=data_path, stage="train", transform=None)

    da_embs_neardiff = da_embs.sel(tile_type="anchor") - da_embs.sel(
        tile_type="neighbor"
    )
    da_embs_neardiff_mag = vector_norm(da_embs_neardiff, dim="emb_dim")

    ds = xr.Dataset(dict(emb=da_embs, an_dist=da_embs_neardiff_mag))

    ds = ds.where(ds.an_dist < 0.1, drop=True)
    print(ds.tile_id.count())

    _save_embeddings(da_embs=ds.emb, dset=dset)


def _save_embeddings(da_embs, dset):
    tile_type = "anchor"
    da_embs = da_embs.sel(tile_type=tile_type)

    # before we get the images we remove all the transforms so that we get the
    # original RGB image
    dset.transform = None
    label_img = []

    img_first_tile = dset[0][0]
    nc, nx, ny = img_first_tile.shape
    ntiles = int(da_embs.tile_id.count())
    label_img = torch.zeros((ntiles, nc, nx, ny))

    for i, tile_id in enumerate(tqdm(da_embs.tile_id.values)):
        # for i, triplet in enumerate(tqdm(dset)):
        label_img[i] = dset[tile_id][0]

    writer = SummaryWriter()
    writer.add_embedding(
        da_embs.transpose("tile_id", "emb_dim").values,
        label_img=label_img,
    )
    writer.close()

    print(
        """
    embeddings saved for tensorboard to `runs/`
    now start tensorboard:
        $> tensorboard --logdir runs
    and open a browser to view the tensorboard embedding projector:
        http://localhost:6006/#projector
    """
    )


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--embs")
    args = argparser.parse_args()
    main2(embs_path=args.embs)


if __name__ == "!!__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    command_group = argparser.add_mutually_exclusive_group()
    command_group.add_argument("--embs")
    command_group.add_argument(
        "--model",
        help=(
            "Path to saved embedding model to use. Pretrained networks"
            " can be used by prefixing with `ex://` for example"
            " `ex://FIXED_NORM_STAGE2`"
        ),
    )
    argparser.add_argument(
        "data",
        help=(
            "Path to tile dataset to use. Example datasets can"
            " be used by prefixing with `ex://` for example `ex://SMALL100`"
        ),
        parents=[None],
    )
    args = argparser.parse_args()
    main(model_path=args.model, data_path=args.data)
