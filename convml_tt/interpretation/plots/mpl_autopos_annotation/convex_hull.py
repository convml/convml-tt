import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull


def _import_matplotlib():
    import matplotlib.pyplot as plt

    return plt


MIN_DISTANCE = 0.1


def _filter_close_points(x, y):
    # if any two neihbouring points are very close we might end up with kinks
    # in the cubic spline, so we filter out those close points here so we end
    # up with a smooth shape
    points = np.array([x, y])
    dl = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=0)

    return x[dl > MIN_DISTANCE], y[dl > MIN_DISTANCE]


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

    x_h, y_h = points[vertices, 0], points[vertices, 1]

    if show_plot:
        plt = _import_matplotlib()
        plt.plot(x_h, y_h, "r--", lw=2)

    def make_t(x, y):
        t = np.arange(x.shape[0], dtype=float)
        t /= t[-1]
        return t

    x_h, y_h = _filter_close_points(x_h, y_h)

    # again we have to ensure that the path is cyclic
    x_h = np.insert(x_h, 0, x_h[-1])
    y_h = np.insert(y_h, 0, y_h[-1])

    t = make_t(x_h, y_h)
    Nt = 100
    nt = np.linspace(0, 1, Nt)

    cs_x = CubicSpline(t, x_h, bc_type="periodic")
    cs_y = CubicSpline(t, y_h, bc_type="periodic")

    x_s = cs_x(nt)
    y_s = cs_y(nt)

    if show_plot:
        plt = _import_matplotlib()
        plt.plot(x_s, y_s, marker=".")

    points_s = np.array([x_s, y_s]).T

    lx, ly = np.max(x_s) - np.min(x_s), np.max(y_s) - np.min(y_s)
    ln = np.sqrt(lx**2.0 + ly**2.0)

    offset_points = []

    for n in range(Np):
        point = points[n]

        dist_xy = point - points_s
        dist_xy[:, 0] /= lx
        dist_xy[:, 1] /= ly

        dists = np.linalg.norm(dist_xy, axis=-1)
        k = np.argmin(dists)
        point_nearest = points_s[k]

        if dists[k] < 0.1:
            if show_plot:
                plt.plot(point_nearest[0], point_nearest[1], marker="s", color="red")

            kl, kr = k - 5, k + 5
            if kr >= Nt:
                kr -= Nt
            d = points_s[kr] - points_s[kl]
            d[0] /= lx
            d[1] /= ly
            d = np.array([d[1], -d[0]])
            d /= np.linalg.norm(d)

            d[0] *= scale * lx
            d[1] *= scale * ly
        else:
            if show_plot:
                (line,) = plt.plot(point_nearest[0], point_nearest[1], marker="s")
                plt.plot(point[0], point[1], marker="o", color=line.get_color())

            d = point_nearest - point
            d /= np.linalg.norm(d)
            d *= scale * ln

        point_outside = point_nearest + d

        if show_plot:
            plt.plot(point_outside[0], point_outside[1], marker=".", color="blue")

        offset_points.append(point_outside)

    return np.array(offset_points)
