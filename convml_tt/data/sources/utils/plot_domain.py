import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import luigi

from .. import DataSource
from ..sampling.domain import SourceDataDomain
from ..pipeline.sampling import SceneSourceFiles
from ..pipeline.scene_sources import GenerateSceneIDs


def _plot_scene_outline(ax, da_scene, scene_num=0, color="orange"):
    x_all, y_all = xr.broadcast(da_scene.x, da_scene.y)

    def border_elems(a, W):
        n1 = a.shape[0]
        r1 = np.minimum(np.arange(n1)[::-1], np.arange(n1))
        n2 = a.shape[1]
        r2 = np.minimum(np.arange(n2)[::-1], np.arange(n2))
        return a[np.minimum(r1[:, None], r2) < W]

    x_edge = border_elems(x_all.values, 1).flatten()
    y_edge = border_elems(y_all.values, 1).flatten()

    return ax.scatter(
        x_edge,
        y_edge,
        transform=da_scene.crs,
        s=1,
        color=color,
        label="source data",
    )


def plot_domain(dataset, ax, **kwargs):
    try:
        ax.gridlines(linestyle="--", draw_labels=True)
    except TypeError:
        ax.gridlines(linestyle="--", draw_labels=False)
    ax.coastlines(resolution="10m", color="grey")

    lines = []

    def draw_box(geom, color, face_alpha=0.5, label=None):
        lines.append(Line2D([0], [0], color=color, lw=1, label=label))
        kwargs = dict(crs=ccrs.PlateCarree(), edgecolor=color)
        ax.add_geometries(
            [
                geom,
            ],
            alpha=face_alpha,
            facecolor=color,
            **kwargs
        )
        ax.add_geometries(
            [
                geom,
            ],
            alpha=face_alpha * 2.0,
            facecolor="none",
            linewidth=1.0,
            label=label,
            **kwargs
        )

    domain = dataset.domain
    if isinstance(domain, SourceDataDomain):
        t_scene_ids = GenerateSceneIDs(data_path=dataset.data_path)
        luigi.build([t_scene_ids], local_scheduler=True)
        sources = t_scene_ids.output().open()
        scene_ids = list(sources.keys())
        scene_id = scene_ids[0]
        t_source_data = SceneSourceFiles(data_path=dataset.data_path, scene_id=scene_id)
        luigi.build([t_source_data], local_scheduler=True)
        da_src = t_source_data.output().open()

        domain = domain.generate_from_dataset(da_src)

    print(domain)
    domain.plot_outline(ax=ax, set_ax_extent=True)
    ax.margins(10.0)
    # bbox_shape = domain_bbox.get_outline_shape()
    # draw_box(bbox_shape, color="red", face_alpha=0.2, label="tiling bbox")

    # domain_rect = None  # TODO: fix to get the correct variable
    # bbox_shape = domain_rect.get_outline_shape()
    # draw_box(bbox_shape, color="green", face_alpha=0.2, label="rect domain")

    # _plot_scene_outline(ax=ax)

    # [x0, x1, y0, y1]
    # ax.legend(lines, [line.get_label() for line in lines])


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", default=".")
    argparser.add_argument("--projection", default="PlateCarree", choices=vars(ccrs))
    args = argparser.parse_args()

    Projection = getattr(ccrs, args.projection)

    dataset = DataSource.load(args.path)
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=Projection()))
    plot_domain(ax=ax, dataset=dataset)
    fn = "domain.png"
    plt.savefig(fn)
    print("Saved domain plot to `{}`".format(fn))
