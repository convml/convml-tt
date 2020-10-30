import base64

import xarray as xr

from faerun import Faerun
import tmap as tm

from .data import IMAGE_TILE_FILENAME_FORMAT, DatasetImagePredictionMapImageTiles
from pathlib import Path
from tqdm import tqdm


def _get_tile_image_path(dataset_path, scene_id, i0, j0):
    t = DatasetImagePredictionMapImageTiles(
        dataset_path=dataset_path,
        scene_id=scene_id,
        step_size=30,
    )

    fn = Path(t.output().fn).parent / IMAGE_TILE_FILENAME_FORMAT.format(i0=i0, j0=j0)
    return fn


def main(da, dataset_path):
    """ Main function """

    # Initialize and configure tmap
    dims = 2048
    enc = tm.Minhash(int(da.emb_dim.count()), 42, dims)
    lf = tm.LSHForest(dims * 2, 128, weighted=True)

    labels = []
    image_labels = []

    data = []

    for tile_id in tqdm(da.tile_id.values):
        da_ = da.sel(tile_id=tile_id)

        i0, j0, scene_id = da_.i0.item(), da_.j0.item(), tile_id[-1]

        labels.append(0)
        data.append(da_.values)

        fn = _get_tile_image_path(dataset_path=dataset_path, scene_id=scene_id, i0=i0, j0=j0)
        img_str = base64.b64encode(open(fn, "rb").read())
        image_labels.append(
            "data:image/png;base64," + str(img_str).replace("b'", "").replace("'", "")
        )

    lf.batch_add(enc.batch_from_weight_array(data))
    lf.index()

    x, y, s, t, _ = tm.layout_from_lsh_forest(lf)

    faerun = Faerun(clear_color="#111111", view="front", coords=False)
    faerun.add_scatter(
        "EMB",
        {"x": x, "y": y, "c": labels, "labels": image_labels},
        colormap="tab20",
        shader="smoothCircle",
        point_scale=10.0,
        max_point_size=50,
        has_legend=True,
        categorical=True,
    )
    faerun.add_tree(
        "EMB_tree", {"from": s, "to": t}, point_helper="EMB", color="#666666"
    )
    faerun.plot("tmap", template="url_image")


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename')
    args = argparser.parse_args()

    da_emb = xr.open_dataarray(args.filename)

    if "scene_id" in da_emb.coords:
        da = da_emb.stack(dict(tile_id=('x', 'y', 'scene_id')))
    else:
        raise NotImplementedError(da_emb.dims)

    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main(da=da, dataset_path=Path("."))
