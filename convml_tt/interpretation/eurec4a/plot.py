from pathlib import Path

import luigi
import skimage.color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr

from ...data.dataset import SceneBulkProcessingBaseTask, TripletDataset
from ...data.sources.satellite import tiler
from ...data.sources.satellite.rectpred import MakeRectRGBImage
from ...pipeline import XArrayTarget
from .data import DatasetImagePredictionMapData
from .transform import DatasetEmbeddingTransform


def _get_img_with_extent_cropped(da_emb, img_fn):
    """
    Load the image in `img_fn`, clip the image and return the
    image extent (xy-extent if the source data coordinates are available)
    """
    img = mpimg.imread(img_fn)

    i_ = da_emb.i0.values
    # NB: j-values might be decreasing in number if we're using a
    # y-coordinate with increasing values from bottom left, so we
    # sort first
    j_ = np.sort(da_emb.j0.values)

    def get_spacing(v):
        dv_all = np.diff(v)
        assert np.all(dv_all[0] == dv_all)
        return dv_all[0]

    i_step = get_spacing(i_)
    j_step = get_spacing(j_)

    ilim = (i_.min() - i_step // 2, i_.max() + i_step // 2)
    jlim = (j_.min() - j_step // 2, j_.max() + j_step // 2)

    img = img[slice(*jlim), slice(*ilim)]

    if "x" in da_emb.coords and "y" in da_emb.coords:
        # get x,y and indexing (i,j) extent from data array so
        # so we can clip and plot the image correctly
        x_min = da_emb.x.min()
        y_min = da_emb.y.min()
        x_max = da_emb.x.max()
        y_max = da_emb.y.max()

        dx = get_spacing(da_emb.x.values)
        dy = get_spacing(da_emb.y.values)
        xlim = (x_min - dx // 2, x_max + dx // 2)
        ylim = (y_min - dy // 2, y_max + dy // 2)

        extent = [*xlim, *ylim]

    return img, np.array(extent)


def _get_img_with_extent(da_emb, img_fn, dataset_path):
    """
    Load the image in `img_fn` and return the
    image extent (xy-extent if the source data coordinates are available)
    """
    img = mpimg.imread(img_fn)

    dataset = TripletDataset.load(dataset_path)
    domain_rect = tiler.RectTile(**dataset.extra["rectpred"]["domain"])
    return img, domain_rect.get_grid_extent()


def _make_rgb(da, dims, alpha=0.5):
    def scale_zero_one(v):
        return (v - v.min()) / (v.max() - v.min())

    scale = scale_zero_one

    d0, d1, d2 = da.dims
    assert d1 == "x" and d2 == "y"

    da_rgba = xr.DataArray(
        np.zeros((4, len(da.x), len(da.y))),
        dims=("rgba", "x", "y"),
        coords=dict(rgba=np.arange(4), x=da.x, y=da.y),
    )

    def _make_component(da_):
        if da_.rgba == 3:
            return alpha * np.ones_like(da_)
        else:
            return scale(da.sel({d0: dims[da_.rgba.item()]}).values)

    da_rgba = da_rgba.groupby("rgba").apply(_make_component)

    return da_rgba


class ComponentsAnnotationMapImage(luigi.Task):
    input_path = luigi.Parameter()
    components = luigi.ListParameter(default=[0, 1, 2])
    src_data_path = luigi.Parameter()
    col_wrap = luigi.IntParameter(default=2)

    def run(self):
        da_emb = xr.open_dataarray(self.input_path)

        da_emb.coords["pca_dim"] = np.arange(da_emb.pca_dim.count())

        da_emb = da_emb.assign_coords(
            x=da_emb.x / 1000.0,
            y=da_emb.y / 1000.0,
            explained_variance=np.round(da_emb.explained_variance, 2),
        )
        da_emb.x.attrs["units"] = "km"
        da_emb.y.attrs["units"] = "km"

        img, img_extent = self.get_image(da_emb=da_emb)

        img_extent = np.array(img_extent) / 1000.0

        # find non-xy dim
        d_not_xy = next(filter(lambda d: d not in ["x", "y"], da_emb.dims))

        N_subplots = len(self.components) + 1
        data_r = 3.0
        ncols = self.col_wrap
        size = 3.0

        nrows = int(np.ceil(N_subplots / ncols))
        figsize = (int(size * data_r * ncols), int(size * nrows))

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=nrows,
            ncols=ncols,
            subplot_kw=dict(aspect=1),
            sharex=True,
        )

        ax = axes.flatten()[0]
        ax.imshow(img, extent=img_extent)
        ax.set_title(da_emb.scene_id.item())

        for n, ax in zip(self.components, axes.flatten()[1:]):
            ax.imshow(img, extent=img_extent)
            da_ = da_emb.sel(**{d_not_xy: n})
            da_ = da_.drop(["i0", "j0", "scene_id"])

            da_.plot.imshow(ax=ax, y="y", alpha=0.5, add_colorbar=False)

            ax.set_xlim(*img_extent[:2])
            ax.set_ylim(*img_extent[2:])

        [ax.set_aspect(1) for ax in axes.flatten()]
        [ax.set_xlabel("") for ax in axes[:-1, :].flatten()]

        plt.tight_layout()

        fig.text(
            0.0,
            -0.02,
            "cum. explained variance: {}".format(
                np.cumsum(da_emb.explained_variance.values)
            ),
        )

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(self.output().fn, bbox_inches="tight")
        plt.close(fig)

    def get_image(self, da_emb):
        raise NotImplementedError

    def output(self):
        image_fullpath = Path(self.input_path)
        src_path, src_fn = image_fullpath.parent, image_fullpath.name

        fn_out = src_fn.replace(
            ".nc",
            ".map.{}__comp.png".format("_".join([str(v) for v in self.components])),
        )

        p = Path(src_path) / fn_out

        return luigi.LocalTarget(str(p))


class DatasetComponentsAnnotationMapImage(ComponentsAnnotationMapImage):
    dataset_path = luigi.Parameter()
    step_size = luigi.Parameter()
    model_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    transform_type = luigi.OptionalParameter()
    transform_extra_args = luigi.OptionalParameter()
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    components = luigi.ListParameter(default=[0, 1, 2])
    crop_img = luigi.BoolParameter(default=False)

    def requires(self):
        if self.transform_type is None:
            return DatasetImagePredictionMapData(
                dataset_path=self.dataset_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
            )
        else:
            return DatasetEmbeddingTransform(
                dataset_path=self.dataset_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
                transform_type=self.transform_type,
                transform_extra_args=self.transform_extra_args,
                pretrained_model=self.pretrained_transform_model,
            )

    @property
    def input_path(self):
        return self.input().fn

    @property
    def src_data_path(self):
        return self.dataset_path

    def get_image(self, da_emb):
        img_path = (
            MakeRectRGBImage(dataset_path=self.dataset_path, scene_id=self.scene_id)
            .output()
            .fn
        )

        if self.crop_img:
            return _get_img_with_extent_cropped(da_emb, img_path)
        else:
            return _get_img_with_extent(
                da_emb=da_emb, img_fn=img_path, dataset_path=self.dataset_path
            )

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "")

        fn = "{}.{}_step.{}_transform.map.{}__comp.png".format(
            self.scene_id,
            self.step_size,
            self.transform_type,
            "_".join([str(v) for v in self.components]),
        )

        p_root = Path(self.dataset_path) / "embeddings" / "rect" / model_name

        if self.pretrained_transform_model is not None:
            p = p_root / self.pretrained_transform_model / "components_map" / fn
        else:
            p = p_root / "components_map" / fn
        return XArrayTarget(str(p))


class AllDatasetComponentAnnotationMapImages(SceneBulkProcessingBaseTask):
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()
    transform_type = luigi.OptionalParameter()
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    components = luigi.ListParameter(default=[0, 1, 2])

    TaskClass = DatasetComponentsAnnotationMapImage

    def _get_task_class_kwargs(self):
        return dict(
            model_path=self.model_path,
            step_size=self.step_size,
            transform_type=self.transform_type,
            pretrained_transform_model=self.pretrained_transform_model,
            components=self.components,
        )


def plot_scene_image(da, dataset_path, crop_image=True, ax=None):
    """
    Render the RGB image of the scene with `scene_id` in `da` into `ax` (a new
    figure will be created if `ax=None`)
    """
    if len(da.shape) == 3:
        # ensure non-xy dim is first
        d_not_xy = list(filter(lambda d: d not in ["x", "y"], da.dims))
        da = da.transpose(*d_not_xy, "x", "y")

    def _get_image():
        if "scene_id" in list(da.coords) + list(da.attrs.keys()):
            try:
                scene_id = da.attrs["scene_id"]
            except KeyError:
                scene_id = da.scene_id.item()
                assert type(scene_id) == str
            img_path = (
                MakeRectRGBImage(dataset_path=dataset_path, scene_id=scene_id)
                .output()
                .fn
            )

            if crop_image:
                return _get_img_with_extent_cropped(da, img_path)
            else:
                return _get_img_with_extent(
                    da, img_fn=img_path, dataset_path=dataset_path
                )
        else:
            raise NotImplementedError(da)

    img, img_extent = _get_image()

    if ax is None:
        lx = img_extent[1] - img_extent[0]
        ly = img_extent[3] - img_extent[2]
        r = lx / ly
        fig_height = 3.0
        fig_width = fig_height * r
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect(1.0)
        ax.set_xlabel(xr.plot.utils.label_from_attrs(da.x))
        ax.set_ylabel(xr.plot.utils.label_from_attrs(da.y))
    ax.imshow(img, extent=img_extent, rasterized=True)

    return ax, img_extent


def make_rgb_annotation_map_image(
    da, rgb_components, dataset_path, render_tiles=False, crop_image=True
):
    """
    Render the contents of `da` onto the RGB image represented by the scene
    (`scene_id` expected to be defined for `da`).

    If `da` is 2D it will be rendered with discrete colours, otherwise if `da`
    is 3D the components of da to use should be given by `rgb_components`
    """
    if len(da.shape) == 3:
        # ensure non-xy dim is first
        d_not_xy = list(filter(lambda d: d not in ["x", "y"], da.dims))
        da = da.transpose(*d_not_xy, "x", "y")

    # now we get to adding the annotation

    # scale distances to km
    if da.x.units == "m" and da.y.units == "m":
        s = 1000.0
        da = da.copy().assign_coords(dict(x=da.x.values / s, y=da.y.values / s))
        da.x.attrs["units"] = "km"
        da.y.attrs["units"] = "km"
    else:
        s = 1.0

    if len(da.shape) == 3:
        da_rgba = _make_rgb(da=da, dims=rgb_components, alpha=0.5)
    elif len(da.shape) == 2:
        # when we have distinct classes (identified by integers) we just
        # want to map each label to a RGB color
        labels = da.stack(dict(n=da.dims))
        arr_rgb = skimage.color.label2rgb(label=labels.values, bg_color=(1.0, 1.0, 1.0))
        # make an RGBA array so we can apply some alpha blending later
        rgba_shape = list(arr_rgb.shape)
        rgba_shape[-1] += 1
        arr_rgba = 0.3 * np.ones((rgba_shape))
        arr_rgba[..., :3] = arr_rgb
        # and put this into a DataArray, unstack to recover original dimensions
        da_rgba = xr.DataArray(
            arr_rgba, dims=("n", "rgba"), coords=dict(n=labels.n)
        ).unstack("n")
    else:
        raise NotImplementedError(da.shape)

    # set up the figure
    nrows = render_tiles and 4 or 3
    fig, axes = plt.subplots(
        figsize=(8, 3.2 * nrows),
        nrows=nrows,
        subplot_kw=dict(aspect=1),
        sharex=True,
    )

    ax = axes[0]
    _, img_extent = plot_scene_image(
        da=da, dataset_path=dataset_path, ax=ax, crop_image=crop_image
    )

    ax = axes[1]
    plot_scene_image(da=da, dataset_path=dataset_path, ax=ax, crop_image=crop_image)
    da_rgba.plot.imshow(ax=ax, rgb="rgba", y="y", rasterized=True)

    ax = axes[2]
    da_rgba[3] = 1.0
    da_rgba.plot.imshow(ax=ax, rgb="rgba", y="y", rasterized=True)

    if "lx_tile" in da.attrs and "ly_tile" in da.attrs:
        if render_tiles:
            x_, y_ = xr.broadcast(da.x, da.y)
            axes[2].scatter(x_, y_, marker="x")

            lx = da.lx_tile / s
            ly = da.ly_tile / s
            ax = axes[3]
            ax.imshow(img, extent=img_extent)
            for xc, yc in zip(x_.values.flatten(), y_.values.flatten()):
                c = da_rgba.sel(x=xc, y=yc).astype(float)
                # set alpha so we can see overlapping tiles
                c[-1] = 0.2
                c_edge = np.array(c)
                c_edge[-1] = 0.5
                rect = mpatches.Rectangle(
                    (xc - lx / 2.0, yc - ly / 2),
                    lx,
                    ly,
                    linewidth=1,
                    edgecolor=c_edge,
                    facecolor=c,
                    linestyle=":",
                )
                ax.scatter(xc, yc, color=c[:-1], marker="x")

                ax.add_patch(rect)

            def pad_lims(lim):
                plen = min(lim[1] - lim[0], lim[3] - lim[2])
                return [
                    (lim[0] - 0.1 * plen, lim[1] + plen * 0.1),
                    (lim[2] - 0.1 * plen, lim[3] + plen * 0.1),
                ]

            xlim, ylim = pad_lims(img_extent)
        else:
            x0, y0 = da.x.min(), da.y.max()
            lx = da.lx_tile / s
            ly = da.ly_tile / s
            rect = mpatches.Rectangle(
                (x0 - lx / 2.0, y0 - ly / 2),
                lx,
                ly,
                linewidth=1,
                edgecolor="grey",
                facecolor="none",
                linestyle=":",
            )
            ax.add_patch(rect)

    xlim = img_extent[:2]
    ylim = img_extent[2:]

    [ax.set_xlim(xlim) for ax in axes]
    [ax.set_ylim(ylim) for ax in axes]
    [ax.set_aspect(1) for ax in axes]

    plt.tight_layout()

    return fig, axes


class RGBAnnotationMapImage(luigi.Task):
    input_path = luigi.Parameter()
    rgb_components = luigi.ListParameter(default=[0, 1, 2])
    src_data_path = luigi.Parameter()
    render_tiles = luigi.BoolParameter(default=False)

    def make_plot(self, da_emb):
        return make_rgb_annotation_map_image(
            da=da_emb,
            rgb_components=self.rgb_components,
            dataset_path=self.dataset_path,
        )

    def run(self):
        da_emb = xr.open_dataarray(self.input_path)
        fig, axes = self.make_plot(da_emb=da_emb)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(self.output().fn, fig=fig, bbox_inches="tight")

    def output(self):
        image_fullpath = Path(self.input_path)
        src_path, src_fn = image_fullpath.parent, image_fullpath.name

        fn_out = src_fn.replace(
            ".nc",
            ".rgb_map.{}__comp.png".format(
                "_".join([str(v) for v in self.rgb_components])
            ),
        )

        p = Path(src_path) / fn_out

        return luigi.LocalTarget(str(p))


class DatasetRGBAnnotationMapImage(RGBAnnotationMapImage):
    dataset_path = luigi.Parameter()
    step_size = luigi.Parameter()
    model_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    transform_type = luigi.OptionalParameter(default=None)
    transform_extra_args = luigi.OptionalParameter(default=None)
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    rgb_components = luigi.ListParameter(default=[0, 1, 2])
    crop_img = luigi.BoolParameter(default=False)

    def requires(self):
        if self.transform_type is None:
            return DatasetImagePredictionMapData(
                dataset_path=self.dataset_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
            )
        else:
            return DatasetEmbeddingTransform(
                dataset_path=self.dataset_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
                transform_type=self.transform_type,
                transform_extra_args=self.transform_extra_args,
                pretrained_model=self.pretrained_transform_model,
                # n_clusters=max(self.rgb_components)+1, TODO: put into transform_extra_args
            )

    def run(self):
        dataset = TripletDataset.load(self.dataset_path)
        da_emb = xr.open_dataarray(self.input_path)

        N_tile = (256, 256)
        model_resolution = da_emb.lx_tile / N_tile[0] / 1000.0
        domain_rect = dataset.extra["rectpred"]["domain"]
        lat0, lon0 = domain_rect["lat0"], domain_rect["lon0"]

        title_parts = [
            self.scene_id,
            "(lat0, lon0)=({}, {})".format(lat0, lon0),
            "{} NN model, {} x {} tiles at {:.2f}km resolution".format(
                self.model_path.replace(".pkl", ""),
                N_tile[0],
                N_tile[1],
                model_resolution,
            ),
            "prediction RGB from {} components [{}]".format(
                da_emb.transform_type, ", ".join([str(v) for v in self.rgb_components])
            ),
        ]
        if self.transform_extra_args:
            title_parts.append(self.requires()._build_transform_identifier())
        title = "\n".join(title_parts)

        fig, axes = self.make_plot(da_emb=da_emb)
        fig.suptitle(title, y=1.05)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(self.output().fn, fig=fig, bbox_inches="tight")

    @property
    def input_path(self):
        return self.input().fn

    @property
    def src_data_path(self):
        return self.dataset_path

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "")

        fn_parts = [
            self.scene_id,
            f"{self.step_size}_step",
            "rgb_map",
            "{}__comp".format("_".join([str(v) for v in self.rgb_components])),
        ]

        if self.transform_type:
            fn_parts.insert(2, self.requires()._build_transform_identifier())

        fn = f"{'.'.join(fn_parts)}.png"

        p_root = Path(self.dataset_path) / "embeddings" / "rect" / model_name
        if self.pretrained_transform_model is not None:
            p = p_root / self.pretrained_transform_model / fn
        else:
            p = p_root / fn
        return XArrayTarget(str(p))


class AllDatasetRGBAnnotationMapImages(SceneBulkProcessingBaseTask):
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()
    transform_type = luigi.OptionalParameter()
    transform_extra_args = luigi.OptionalParameter(default=None)
    pretrained_transform_model = luigi.OptionalParameter(default=None)
    rgb_components = luigi.ListParameter(default=[0, 1, 2])

    TaskClass = DatasetRGBAnnotationMapImage

    def _get_task_class_kwargs(self):
        return dict(
            model_path=self.model_path,
            step_size=self.step_size,
            transform_type=self.transform_type,
            transform_extra_args=self.transform_extra_args,
            pretrained_transform_model=self.pretrained_transform_model,
            rgb_components=self.rgb_components,
        )
