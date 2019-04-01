import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as il

from ..utils import get_triplets_from_encodings


from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline

def calc_point_offsets(points, scale=0.2, show_plot=False):
    """
    Calculate offset point for each point in points which is outside a smooth
    rerepresentation of the convex hull of points in 2D. This is useful for
    positioning labels or inset axes outside plotted points
    """
    Np, _ = points.shape
    hull = ConvexHull(points)

    vertices = list(hull.vertices)
    vertices.insert(0, vertices[-1])

    x_h, y_h = points[vertices,0], points[vertices,1]

    if show_plot:
        plt.plot(x_h, y_h, 'r--', lw=2)

    def make_t(x, y):
        t = np.arange(x.shape[0], dtype=float)
        t /= t[-1]
        return t

    t = make_t(x_h, y_h)
    nt = np.linspace(0, 1, 100)

    cs_x = CubicSpline(t, x_h, bc_type='periodic')
    cs_y = CubicSpline(t, y_h, bc_type='periodic')

    x_s = cs_x(nt)
    y_s = cs_y(nt)

    points_s = np.array([x_s, y_s]).T

    lx, ly = np.max(x_s)-np.min(x_s), np.max(y_s)-np.min(y_s)

    offset_points = []

    for n in range(Np):
        point = points[n]

        dist_xy = point - points_s
        dist_xy[:,0] /= lx
        dist_xy[:,1] /= ly

        dists = np.linalg.norm(dist_xy, axis=-1)
        k = np.argmin(dists)

        point_nearest = points_s[k]

        if show_plot:
            plt.plot(point_nearest[0], point_nearest[1], marker='s', color='red')

        d = points_s[k+1] - points_s[k-1]
        d[0] /= lx
        d[1] /= ly
        d = np.array([d[1], -d[0]])
        d /= np.linalg.norm(d)

        d[0] *= scale*lx
        d[1] *= scale*ly

        point_outside = point_nearest + d

        if show_plot:
            plt.plot(point_outside[0], point_outside[1], marker='.', color='blue')

        offset_points.append(point_outside)

    return np.array(offset_points)


def plot_encoding_properties_with_examples(x, y, points, ax=None, size=0.2):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    else:
        fig = ax.figure

    triplets = get_triplets_from_encodings(x)

    if type(points) == int:
        N = len(x)
        idx_sample = np.random.choice(np.arange(N), size=points)
        x_sample, y_sample = x[idx_sample], y[idx_sample]
    elif type(points) == np.ndarray:
        x_sample = x.sel(tile_id=points)
        y_sample = y.sel(tile_id=points)
    else:
        raise NotImplementedError(type(points))
        
    ax.scatter(x, y, marker='.', alpha=0.2, color='grey')

    ax.scatter(x_sample, y_sample, marker='.')

    ax.set_xlabel(x._title_for_slice())
    ax.set_ylabel(y._title_for_slice())

    pts = np.array([x_sample, y_sample]).T
    pts_offset = calc_point_offsets(pts, scale=3*size)

    def transform(coord):
        return (ax.transData + fig.transFigure.inverted()).transform(coord)

    for n, tile_id in enumerate(x_sample.tile_id):
        x_, y_ = pts_offset[n]
        #tile_id, color = tile_ids[n]

        pts_connector = np.c_[pts_offset[n], pts[n]]
        ax.plot(*pts_connector, linestyle='--', alpha=0.5)

        xp, yh = transform((x_, y_))

        ax1=fig.add_axes([xp-0.5*size, yh-size*0.5, size, size])
        ax1.set_aspect(1)
        ax1.axison = False

        img_idx = int(tile_id.values)
        img = triplets[img_idx]

        # img = Image.open(tiles_path/"{:05d}_anchor.png".format())
        ax1.imshow(img)
