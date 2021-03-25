#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import sklearn.decomposition
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy as hc

from ...data.dataset import ImageSingletDataset, TileType


# In[13]:


def _make_letter_labels(n_labels):
    return np.array([chr(i + 97).upper() for i in np.arange(n_labels)])


def _fix_labels(ax, leaf_mapping, label_clusters=False):
    """
    Initially the labels simply correspond the leaf index, but we want to plot
    the number of items that belong to that leaf. And optionally give each leaf
    node a label (A, B, C, etc)
    """
    new_labels = []

    for (i, label) in enumerate(ax.get_xticklabels()):
        leaf_index = int(label.get_text())
        num_items_in_leaf = np.sum(leaf_index == leaf_mapping)
        new_label = str(num_items_in_leaf)
        if label_clusters:
            new_label += "\n{}".format(chr(i + 97).upper())
        new_labels.append(new_label)

    return new_labels


def dendrogram(
    da_embeddings,
    n_clusters_max=14,
    debug=False,
    ax=None,
    n_samples=10,
    show_legend=False,
    label_clusters=False,
    return_clusters=False,
    color="black",
    sampling_method="random",
    linkage_method="ward",
    **kwargs
):
    """
    Additional kwargs will be passed to scipy.cluster.hierarchy.dendrogram
    """

    tile_dataset = ImageSingletDataset(
        data_dir=da_embeddings.data_dir,
        tile_type=da_embeddings.tile_type,
        stage=da_embeddings.stage,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))
    else:
        fig = ax.figure

    Z = hc.linkage(
        y=da_embeddings,
        method=linkage_method,
    )

    if color is not None:
        kwargs["link_color_func"] = lambda k: color

    # we want to label the leaf by the index of the leaf node, at least
    # initially. Below we will change the labels to have the count in each
    # leaf, but we don't know that number yet
    leaf_label_func = lambda i: str(i)
    kwargs["leaf_label_func"] = leaf_label_func

    ddata = hc.dendrogram(
        Z=Z, truncate_mode="lastp", p=n_clusters_max, get_leaves=True, **kwargs
    )

    if debug:
        for ii in range(len(ddata["icoord"])):

            bl, br = list(zip(ddata["icoord"][ii], ddata["dcoord"][ii]))[
                0::3
            ]  # second and third are top left and right corners
            ax.scatter(*bl, marker="s", label=ii, s=100)
            ax.scatter(*br, marker="s", label=ii, s=100)

    # find the lowest link
    y_merges = np.array(ddata["dcoord"])
    d_max = np.min(y_merges[y_merges > 0.0])

    d2 = Z[:, 2][np.argwhere(Z[:, 2] == d_max)[0][0] - 1]

    if debug:
        plt.axhline(d_max, linestyle="--", color="grey")
        ax.legend()

    T = hc.fcluster(Z, t=d2, criterion="distance")
    L, M = hc.leaders(Z, T)

    assert set(L) == set(ddata["leaves"])

    # getting leaf locations
    # the order in `L` (leaders) above unfortunately is *not* same as the order
    # of points in icoord so instead we pick up the order from the actual
    # labels used
    bl_pts = np.array(
        [
            np.asarray(ddata["icoord"])[:, 0],  # x at bottom-right corner
            np.asarray(ddata["dcoord"])[:, 0],  # y at bottom-right corner
        ]
    )
    br_pts = np.array(
        [
            np.asarray(ddata["icoord"])[:, -1],  # x at bottom-right corner
            np.asarray(ddata["dcoord"])[:, -1],  # y at bottom-right corner
        ]
    )

    leaf_pts = np.append(bl_pts, br_pts, axis=1)
    # remove pts where y != 0 as these mark joins within the diagram and don't
    # connect to the edge
    leaf_pts = leaf_pts[:, ~(leaf_pts[1] > 0)]
    # sort by x-coordinate for leaf labels, so that the positions are in the
    # same order as the axis labels
    leaf_pts = leaf_pts[:, leaf_pts[0, :].argsort()]
    # get the actual leaf from the indecies (these were set by providing the
    # `leaf_label_func` above)
    leaf_indecies_from_labels = np.array(
        [int(lab.get_text()) for lab in ax.get_xticklabels()]
    )
    # create mapping from the leaf indecies to the (x,y)-points in the
    # dendrogram where these leaves terminate
    leaf_pts_mapping = dict(zip(leaf_indecies_from_labels, leaf_pts.T))

    # work out which leaf each item (image) belongs to
    mapping = dict(zip(M, L))
    leaf_mapping = np.array(list(map(lambda i: mapping[i], T)))

    N_leaves = len(np.unique(leaf_mapping))

    # counts per leaf
    # [(n, sum(leaf_mapping == n)) for n in L]

    w_pad = 0.02
    size = (3.6 - (n_clusters_max - 1.0) * w_pad) / float(n_clusters_max)
    y_offset = 1.4
    if label_clusters:
        y_offset += 0.2

    for lid, leaf_id in enumerate(ddata["leaves"]):
        img_idxs_in_cluster = da_embeddings.tile_id.values[
            leaf_mapping == leaf_id
        ].astype(int)
        if sampling_method == "random":
            try:
                img_idxs = np.random.choice(
                    img_idxs_in_cluster, size=n_samples, replace=False
                )
            except ValueError:
                img_idxs = img_idxs_in_cluster
        elif sampling_method == "center_dist":
            emb_in_cluster = da_embeddings.sel(tile_id=img_idxs_in_cluster)
            d_emb = emb_in_cluster.mean(dim="tile_id") - emb_in_cluster
            center_dist = np.sqrt(d_emb ** 2.0).sum(dim="emb_dim")
            emb_in_cluster["dist_to_center"] = center_dist
            img_idxs = emb_in_cluster.sortby("dist_to_center").tile_id.values[
                :n_samples
            ]
        else:
            raise NotImplementedError(sampling_method)

        def transform(coord):
            axis_to_data = fig.transFigure + ax.transData.inverted()
            data_to_axis = axis_to_data.inverted()
            return data_to_axis.transform(coord)

        leaf_xy = leaf_pts_mapping[leaf_id]
        xp, yh = transform(leaf_xy)

        if show_legend:
            ax.scatter(*leaf_xy, marker="s", label=lid, s=100)

        for n, img_idx in enumerate(img_idxs):
            img = tile_dataset.get_image(index=img_idx)

            ax1 = fig.add_axes(
                [xp - 0.5 * size, yh - size * 1.1 * (n + y_offset), size, size]
            )
            ax1.set_aspect(1)
            ax1.axison = False
            ax1.imshow(img)

    ax.set_xticklabels(
        _fix_labels(ax=ax, leaf_mapping=leaf_mapping, label_clusters=label_clusters)
    )

    if show_legend:
        ax.legend()

    if return_clusters:
        # instead of returning the actual indecies of the leaves here (as were
        # used above) we remap so that they run from 0...N_leaves
        leaf_idxs_remapped = np.array(
            [list(leaf_indecies_from_labels).index(i) for i in leaf_mapping]
        )
        if not label_clusters:
            return ax, leaf_idxs_remapped
        else:
            return ax, _make_letter_labels(N_leaves)[leaf_idxs_remapped]
    else:
        return ax
