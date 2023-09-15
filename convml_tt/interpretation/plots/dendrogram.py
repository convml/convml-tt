#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.cluster import hierarchy as hc

from ...data.dataset import ImageSingletDataset, ImageTripletDataset, TileType


def _make_letter_labels(n_labels):
    return np.array([chr(i + 97).upper() for i in np.arange(n_labels)])


def _fix_labels(ax, tile_idxs_per_cluster, label_clusters=False):
    """
    Initially the labels simply correspond the leaf index, but we want to plot
    the number of items that belong to that leaf. And optionally give each leaf
    node a label (A, B, C, etc)
    """
    new_labels = []

    for i, cluster_label in enumerate(ax.get_xticklabels()):
        cluster_id = int(cluster_label.get_text())
        items_cluster = tile_idxs_per_cluster[cluster_id]
        num_items_in_leaf = len(items_cluster)
        new_label = str(num_items_in_leaf)
        if label_clusters:
            new_label += "\n{}".format(chr(i + 97).upper())
        new_labels.append(new_label)

    return new_labels


def _get_tile_image(tile_dataset, tile_id, tile_type):
    if isinstance(tile_dataset, ImageTripletDataset):
        tile_image = tile_dataset.get_image(
            tile_id=tile_id, tile_type=TileType[tile_type.upper()]
        )
    elif isinstance(tile_dataset, ImageSingletDataset):
        tile_image = tile_dataset.get_image(tile_id=tile_id)
    else:
        raise NotImplementedError(tile_dataset)
    return tile_image


def _find_tile_indecies(
    tile_idxs_in_cluster, n_samples, sampling_method, da_clustering, da_embeddings
):
    if sampling_method == "random":
        try:
            tile_idxs = np.random.choice(
                tile_idxs_in_cluster, size=n_samples, replace=False
            )
        except ValueError:
            tile_idxs = tile_idxs_in_cluster
    elif sampling_method == "center_dist":
        emb_in_cluster = da_clustering.sel(tile_id=tile_idxs_in_cluster)
        d_emb = emb_in_cluster.mean(dim="tile_id") - emb_in_cluster
        center_dist = np.sqrt(d_emb**2.0).sum(dim="emb_dim")
        emb_in_cluster["dist_to_center"] = center_dist
        tile_idxs = emb_in_cluster.sortby("dist_to_center").tile_id.values[:n_samples]
    elif sampling_method in ["best_triplets", "worst_triplets"]:
        if "tile_type" not in da_embeddings.coords:
            raise Exception(
                "Selection method based on triplets can only be used when"
                " passing in embeddings for triplets"
            )
        da_emb_in_cluster = da_embeddings.sel(tile_id=tile_idxs_in_cluster)
        da_emb_dist = da_emb_in_cluster.sel(tile_type="anchor") - da_emb_in_cluster.sel(
            tile_type="neighbor"
        )
        near_dist = np.sqrt(da_emb_dist**2.0).sum(dim="emb_dim")
        da_emb_in_cluster["near_dist"] = near_dist
        img_idxs_all = da_emb_in_cluster.sortby("near_dist").tile_id.values
        if sampling_method == "best_triplets":
            tile_idxs = img_idxs_all[:n_samples]
        else:
            tile_idxs = img_idxs_all[::-1][:n_samples]
    elif sampling_method == "worst_triplets":
        da_emb_in_cluster = da_embeddings.sel(tile_id=tile_idxs_in_cluster)
        da_emb_dist = da_emb_in_cluster.sel(tile_type="anchor") - da_emb_in_cluster.sel(
            tile_type="neighbor"
        )
        near_dist = np.sqrt(da_emb_dist**2.0).sum(dim="emb_dim")
        da_emb_in_cluster["near_dist"] = near_dist
        tile_idxs = da_emb_in_cluster.sortby("near_dist", reverse=True).tile_id.values[
            :n_samples
        ]
    else:
        raise NotImplementedError(sampling_method)

    return tile_idxs


def _find_leaf_indxs_and_fig_posns(Z, ddata, ax):
    """
    Plot a dendrogram into axes `ax` and return for each leaf cluster the
    item-indecies that belong to that cluster
    """
    # find the lowest link
    y_merges = np.array(ddata["dcoord"])
    d_max = np.min(y_merges[y_merges > 0.0])

    d2 = Z[:, 2][np.argwhere(Z[:, 2] == d_max)[0][0] - 1]

    T = hc.fcluster(Z, t=d2, criterion="distance")
    L, M = hc.leaders(Z, T)

    assert set(L) == set(ddata["leaves"])

    # get the actual leaf from the indecies (these were set by providing the
    # `leaf_label_func` above)
    leaf_indecies_from_labels = np.array(
        [int(lab.get_text()) for lab in ax.get_xticklabels()]
    ).tolist()

    ax.set_xticklabels(np.arange(len(leaf_indecies_from_labels)))

    # work out which leaf each item (image) belongs to
    mapping = dict(zip(M, L))
    leaf_mapping = np.array(list(map(lambda i: mapping[i], T)))

    # counts per leaf
    # [(n, sum(leaf_mapping == n)) for n in L]

    tile_idxs_per_cluster = {}
    for tile_id, leaf_id in enumerate(leaf_mapping):
        cluster_id = leaf_indecies_from_labels.index(leaf_id)
        cluster_tile_idxs = tile_idxs_per_cluster.setdefault(cluster_id, [])
        cluster_tile_idxs.append(tile_id)

    return tile_idxs_per_cluster


def dendrogram(
    da_embeddings,
    n_samples=10,
    n_clusters_max=14,
    sampling_method="random",
    debug=False,
    show_legend=False,
    label_clusters=False,
    return_clusters=False,
    color="black",
    linkage_method="ward",
    **kwargs
):
    """
    Create a dendrogram plot representing the clustering with the embedding
    vectors `da_embddings` with sample tiles below each leaf node of the
    dendrogram. Parameters:

    n_clusters_max:
        number of clusters to terminate the clustering at
    n_samples:
        number of tiles to show for each cluster
    sampling_method:
        method by which the sample tiles which are shown are selected.
        `random`: select the tiles at random from each cluster.
        `center_dist`: select the tiles closest (by Eucledian distance) to the
            cluster centroid in the embedding space.
        `best_triplets`: sort tiles from triplets for which the anchor-neighbor
            distance is closest in the embedding space
        `worst_triplets`: sort tiles from triplets for which the anchor-neighbor
            distance is largest in the embedding space
    show_legend:
        add a colour-legend for the different cluster IDs
    label_clusters:
        create letter-based cluster identifiers (A, B, C...) and put these on
        the figure
    return_clusters:
        return the associated tile ids associated with each cluster in the
        branch nodes of the dendrogram as an xarray.DataArray
    color:
        change the colour of the dendrogram lines

    Additional kwargs will be passed to scipy.cluster.hierarchy.dendrogram
    """
    tile_type = None
    if "tile_type" in da_embeddings.coords:
        embeddings_tts = set(da_embeddings.tile_type.values)
        if set([tt.name.lower() for tt in TileType]) == embeddings_tts:
            tile_dataset = ImageTripletDataset(
                data_dir=da_embeddings.data_dir,
                stage=da_embeddings.stage,
            )
        else:
            raise NotImplementedError(embeddings_tts)
        if "tile_type" not in kwargs:
            raise Exception(
                "You must provide the tile_type when plotting a dendrogram"
                " from embeddings of triplets"
            )
        tile_type = kwargs.pop("tile_type")
    else:
        tile_dataset = ImageSingletDataset(
            data_dir=da_embeddings.data_dir,
            tile_type=da_embeddings.tile_type,
            stage=da_embeddings.stage,
        )
        tile_type = da_embeddings.tile_type

    fig_margin = 0.4  # [inches]
    fig_width = 10.0 + 2.0 * fig_margin  # [inches]
    tile_space = (fig_width - 2.0 * fig_margin) / float(n_clusters_max)

    dendrogram_ny = 2  # in units of tile size
    dendrogram_height = tile_space * dendrogram_ny
    tile_samples_height = tile_space * n_samples
    fig_ny = dendrogram_ny + n_samples * 1
    # bottom margin will be twice the top to make sure the rendered tiles don't
    # get clipped off
    fig_height = dendrogram_height + tile_samples_height + fig_margin * 3.0

    fig = plt.figure(figsize=(fig_width, fig_height))
    gridspec = fig.add_gridspec(ncols=1, nrows=fig_ny)
    ax_dendrogram = fig.add_subplot(gridspec[:dendrogram_ny])

    tile_size = 0.9
    y_offset = tile_size / 2.0 + 0.4 * fig_ny / fig_height  # 0.4 will be in inches
    if label_clusters:
        y_offset += 0.2 * fig_ny / fig_height
    tile_ymax = n_samples + y_offset
    ax_tiles = fig.add_subplot(gridspec[dendrogram_ny:])
    ax_tiles.set_ylim(y_offset, tile_ymax)

    if not debug:
        ax_tiles.axis("off")

    fig.subplots_adjust(
        left=fig_margin / fig_width,
        right=1.0 - fig_margin / fig_width,
        top=1.0 - fig_margin / fig_height,
        bottom=fig_margin / fig_height,
        hspace=0.0,
    )

    if "tile_type" in da_embeddings.coords:
        da_clustering = da_embeddings.sel(tile_type=tile_type)
    else:
        da_clustering = da_embeddings

    Z = hc.linkage(
        y=da_clustering,
        method=linkage_method,
    )

    if color is not None:
        kwargs["link_color_func"] = lambda k: color

    # we want to inititally use the `leaf_label_func` to label the leaf by the
    # index of the leaf node so we can retrieve these indecies.
    # Below we will change the labels to have the count in each leaf, but we
    # don't know that number yet
    def leaf_label_func(i):
        return str(i)

    kwargs["leaf_label_func"] = leaf_label_func

    ddata = hc.dendrogram(
        Z=Z,
        truncate_mode="lastp",
        p=n_clusters_max,
        ax=ax_dendrogram,
        get_leaves=True,
        **kwargs
    )

    # the indecies returned when finding the leaf indecies below number from
    # zero, but we want the actual tile IDs and so need to map into those here
    idxs_per_cluster = _find_leaf_indxs_and_fig_posns(
        Z=Z, ddata=ddata, ax=ax_dendrogram
    )

    tile_idxs_per_cluster = {}
    for c_id, idxs in idxs_per_cluster.items():
        tile_idxs_per_cluster[c_id] = da_embeddings.isel(tile_id=idxs).tile_id.values

    # the dendrogram plot uses the below transform for how to position the leaf
    # cluster points

    def xposn2clusterid(x):
        return (x - 5) / 10

    def clusterid2xposn(x):
        return x * 10 + 5

    # create a secondary axes on the dendrogram axes which is scaled so that
    # the x-positions match the cluster indecies (0, 1, ... n_max_clusters-1)
    # we can then join this with the axes below where we will render the tiles
    # and the x-positioning becomes much easier
    secax = ax_dendrogram.secondary_xaxis(
        "bottom", functions=(xposn2clusterid, clusterid2xposn)
    )
    secax.set_xticklabels([])
    secax.set_xticks([])
    ax_tiles.sharex(secax)

    N_clusters = len(tile_idxs_per_cluster)

    for cluster_id in np.arange(N_clusters):
        tile_idxs_in_cluster = tile_idxs_per_cluster[cluster_id]

        tile_ids = _find_tile_indecies(
            tile_idxs_in_cluster=tile_idxs_in_cluster,
            n_samples=n_samples,
            sampling_method=sampling_method,
            da_clustering=da_clustering,
            da_embeddings=da_embeddings,
        )

        def transform(coord):
            axis_to_data = fig.transFigure + ax_dendrogram.transData.inverted()
            axis_to_data = ax_dendrogram.transData.inverted()
            data_to_axis = axis_to_data.inverted()
            return data_to_axis.transform(coord)

        leaf_xy = [cluster_id, tile_ymax]
        tile_start_xy = [cluster_id, n_samples]

        if show_legend:
            ax_tiles.scatter(*leaf_xy, marker="s", label=cluster_id, s=100)

        for n, img_idx in enumerate(tile_ids):
            img = _get_tile_image(
                tile_dataset=tile_dataset, tile_id=img_idx, tile_type=tile_type
            )

            xp, yh = tile_start_xy
            ax_tile = ax_tiles.inset_axes(
                [
                    xp - 0.5 * tile_size,
                    yh - 1.0 * n - 0.5 * tile_size,
                    tile_size,
                    tile_size,
                ],
                transform=ax_tiles.transData,
            )

            ax_tile.set_aspect(1)
            ax_tile.axison = False
            ax_tile.imshow(img)

    ax_dendrogram.set_xticklabels(
        _fix_labels(
            ax=ax_dendrogram,
            tile_idxs_per_cluster=tile_idxs_per_cluster,
            label_clusters=label_clusters,
        )
    )

    if show_legend:
        ax_tiles.legend()

    if return_clusters:
        arr_clusters = np.empty(int(da_embeddings.tile_id.count()))
        if label_clusters:
            cluster_labels = _make_letter_labels(N_clusters)
            tile_idxs_per_cluster = dict(
                [
                    (cluster_labels[c_id], tile_ids)
                    for (c_id, tile_ids) in tile_idxs_per_cluster.items()
                ]
            )

        label_type = type(list(tile_idxs_per_cluster.keys())[0])
        arr_clusters = np.empty(int(da_embeddings.tile_id.count()), dtype=label_type)
        for c_label, c_tile_ids in tile_idxs_per_cluster.items():
            for t_id in c_tile_ids:
                arr_clusters[t_id] = c_label

        da_clusters = xr.DataArray(
            arr_clusters,
            dims=("tile_id"),
            coords=dict(tile_id=da_embeddings.tile_id),
        )

        return fig, da_clusters
    else:
        return fig
