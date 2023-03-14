import pickle
from pathlib import Path

import xarray as xr
from faerun import Faerun, host
from tqdm import tqdm

from .data import IMAGE_TILE_FILENAME_FORMAT, DatasetImagePredictionMapImageTiles


def _get_tile_image_path(dataset_path, scene_id, i0, j0):
    t = DatasetImagePredictionMapImageTiles(
        dataset_path=dataset_path,
        scene_id=scene_id,
        step_size=50,
    )

    fn = Path(t.output().fn).parent / IMAGE_TILE_FILENAME_FORMAT.format(i0=i0, j0=j0)
    return fn


def main(da, dataset_path):
    """Main function"""

    # Initialize and configure tmap
    # dims = 2048
    # enc = tm.Minhash(int(da.emb_dim.count()), 42, dims)
    # lf = tm.LSHForest(dims * 4, 256, weighted=True)
    # enc = tm.Minhash()
    # lf = tm.LSHForest(weighted=True)

    # labels = []
    image_labels = []

    # data = []

    # da = da.isel(scene_id=0).stack(dict(tile_id=('x', 'y')))
    da = da.sel(x=slice(-1000e3, None)).stack(dict(tile_id=("x", "y", "scene_id")))
    # sel(tile_id=da.tile_id.values[:100])

    # scene_ids = list(da.unstack().scene_id.values)

    # x_ = []
    # y_ = []
    # z_ = []

    for tile_id in tqdm(da.tile_id.values):
        try:
            da_ = da.sel(tile_id=tile_id)
            i0, j0 = da_.i0.item(), da_.j0.item()
            try:
                scene_id = da_.scene_id.item()
            except Exception:
                scene_id = tile_id[-1]
            # x_.append(tile_id[0])
            # y_.append(tile_id[1])

            fn = _get_tile_image_path(
                dataset_path=dataset_path, scene_id=scene_id, i0=i0, j0=j0
            )
            # img_str = base64.b64encode(open(fn, "rb").read())
            # "data:image/png;base64," + str(img_str).replace("b'", "").replace("'", "")

            # labels.append(scene_ids.index(scene_id))
            # c = da_.isel(pca_dim=3).values
            # c = 0
            # labels.append(c)

            image_labels.append("get_image?path=" + str(fn))

            # data.append(da_.values)

        except FileNotFoundError:
            continue

    # lf.batch_add(enc.batch_from_weight_array(data))
    # lf.index()

    # x, y, s, t, _ = tm.layout_from_lsh_forest(lf)

    # x = np.array(x_)
    # y = np.array(y_)
    # z = np.array(z_)

    x = da.isel(pca_dim=0).values
    y = da.isel(pca_dim=1).values
    z = da.isel(pca_dim=2).values
    c = da.isel(pca_dim=3).values

    # from sklearn.manifold import TSNE
    # X = da.values.T
    # X_embedded = TSNE(n_components=2,).fit_transform(X)
    # x, y = X[:,0], X[:,1]

    f = Faerun(
        clear_color="#111111",
        view="front",
        coords=True,
        x_title="pca dim 0",
        y_title="pca dim 1",
    )
    f.add_scatter(
        "EMB",
        {"x": x, "y": y, "z": z, "c": c, "labels": image_labels},
        colormap="jet",
        # shader="smoothCircle",
        shader="sphere",
        point_scale=5.0,
        max_point_size=10,
        has_legend=True,
        categorical=False,
    )
    # f.add_tree(
    # "EMB_tree", {"from": s, "to": t}, point_helper="EMB", color="#666666"
    # )

    with open("tmap.faerun", "wb+") as handle:
        pickle.dump(f.create_python_data(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    # faerun.plot("tmap", template="url_image")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("filename")
    argparser.add_argument("--view", default=False, action="store_true")
    args = argparser.parse_args()

    if not Path("tmap.faerun").exists():
        da = xr.open_dataarray(args.filename)

        # if "scene_id" in da_emb.coords:
        # da = da_emb.stack(dict(tile_id=('x', 'y', 'scene_id')))
        # else:
        # raise NotImplementedError(da_emb.dims)

        import ipdb

        with ipdb.launch_ipdb_on_exception():
            main(da=da, dataset_path=Path("."))

    if args.view:
        host("tmap.faerun", label_type="url_image", theme="dark", view="free")
