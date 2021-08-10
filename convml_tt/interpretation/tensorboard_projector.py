from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import xarray as xr
import torch
import hashlib

from convml_tt.data.dataset import ImageTripletDataset
from convml_tt.data.transforms import get_transforms
from convml_tt.utils import get_embeddings
from convml_tt.system import TripletTrainerModel
from convml_tt.data.examples import (
    fetch_example_dataset,
    ExampleData,
    load_pretrained_model,
    PretrainedModel,
)


def _make_hash(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def main(model_path, data_path):
    if model_path.startswith("ex://"):
        model_name = model_path[5:]
        available_models = [m.name for m in list(PretrainedModel)]
        if model_name not in available_models:
            raise Exception(
                f"Pretrained model `{model_name}` not found."
                f" Available models: {', '.join(available_models)}"
            )
        model = load_pretrained_model(PretrainedModel[model_name])
    else:
        model = TripletTrainerModel.load_from_checkpoint(model_path)

    embs_id = _make_hash(f"{model_path}__{data_path}")
    fpath_embs = Path(f"embs-{embs_id}.nc")

    if data_path.startswith("ex://"):
        data_path = fetch_example_dataset(dataset=ExampleData.SMALL100)
    transforms = get_transforms(step="predict", normalize_for_arch=model.base_arch)
    dset = ImageTripletDataset(data_dir=data_path, stage="train", transform=transforms)

    if not fpath_embs.exists():
        da_embs = get_embeddings(tile_dataset=dset, model=model)
        da_embs.to_netcdf(fpath_embs)
    else:
        da_embs = xr.open_dataarray(fpath_embs)
        print(f"Using saved embeddings from `{fpath_embs}`")

    tile_type = "anchor"
    da_embs = da_embs.sel(tile_type=tile_type)

    # before we get the images we remove all the transforms so that we get the
    # original RGB image
    dset.transform = None
    label_img = []
    for triplet in dset:
        label_img.append(triplet[0])

    label_img = torch.stack(label_img)

    writer = SummaryWriter()
    writer.add_embedding(
        da_embs.transpose("tile_id", "emb_dim").values, label_img=label_img
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
    argparser.add_argument(
        "model",
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
    )
    args = argparser.parse_args()
    main(model_path=args.model, data_path=args.data)
