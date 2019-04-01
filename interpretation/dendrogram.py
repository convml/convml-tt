#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import sklearn.decomposition
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy as hc

from ..utils import get_triplets_from_encodings


# In[13]:

def dendrogram(encodings, n_clusters_max=14, debug=False, ax=None,
    n_samples=10, show_legend=False):

    triplets = get_triplets_from_encodings(encodings)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14,3))
    else:
        fig = ax.figure

    Z = hc.linkage(y=encodings, method='ward',)

    ddata = hc.dendrogram(Z=Z, truncate_mode='lastp', p=n_clusters_max, get_leaves=True)

    if debug:
        for ii in range(len(ddata['icoord'])):

            bl, br = list(zip(ddata['icoord'][ii], ddata['dcoord'][ii]))[0::3] # second and third are top left and right corners
            ax.scatter(*bl, marker='s', label=ii, s=100)
            ax.scatter(*br, marker='s', label=ii, s=100)

    # find the lowest link
    y_merges = np.array(ddata['dcoord'])
    d_max = np.min(y_merges[y_merges > 0.])

    d2 = Z[:,2][np.argwhere(Z[:,2] == d_max)[0][0]-1]

    if debug:
        plt.axhline(d_max, linestyle='--', color='grey')
        ax.legend()

    T = hc.fcluster(Z, t=d2, criterion='distance')
    L, M = hc.leaders(Z, T)

    assert(set(L) == set(ddata['leaves']))

    # getting leaf locations
    # the order in `L` (leaders) above is the same as the order of points in icoord
    bl_pts = np.array([
        np.asarray(ddata['icoord'])[:,0], # x at bottom-right corner
        np.asarray(ddata['dcoord'])[:,0]  # y at bottom-right corner
    ])
    br_pts = np.array([
        np.asarray(ddata['icoord'])[:,-1], # x at bottom-right corner
        np.asarray(ddata['dcoord'])[:,-1]  # y at bottom-right corner
    ])

    leaf_pts = np.append(bl_pts, br_pts, axis=1)
    # remove pts where y != 0 as these mark joins within the diagram and don't connect to the edge
    leaf_pts = leaf_pts[:,~(leaf_pts[1] > 0)]

    leaf_pts_mapping = dict(zip(L, leaf_pts.T))

    assert len(leaf_pts_mapping) == len(ddata['leaves'])

    # work out which leaf each item (image) belongs to
    mapping = dict(zip(M, L))
    leaf_mapping = np.array(list(map(lambda i: mapping[i], T)))
    leaf_mapping

    # counts per leaf
    # [(n, sum(leaf_mapping == n)) for n in L]

    w_pad = 0.02
    size = (3.6 - (n_clusters_max - 1.)*w_pad)/float(n_clusters_max)

    for lid, leaf_id in enumerate(ddata['leaves']):
        img_idxs_in_cluster = encodings.tile_id.values[leaf_mapping == leaf_id].astype(int)
        try:
            img_idxs = np.random.choice(img_idxs_in_cluster, size=n_samples, replace=False)
        except ValueError:
            img_idxs = img_idxs_in_cluster

        def transform(coord):
            axis_to_data = fig.transFigure + ax.transData.inverted()
            data_to_axis = axis_to_data.inverted()
            return data_to_axis.transform(coord)

        leaf_xy = leaf_pts_mapping[leaf_id]
        xp, yh = transform(leaf_xy)

        if show_legend:
            ax.scatter(*leaf_xy, marker='s', label=lid, s=100)

        for n, img_idx in enumerate(img_idxs):

            img = triplets[img_idx]

            ax1=fig.add_axes([xp-0.5*size, yh-size*1.1*(n + 1.4), size, size])
            ax1.set_aspect(1)
            ax1.axison = False
            ax1.imshow(img)

    if show_legend:
        ax.legend()
