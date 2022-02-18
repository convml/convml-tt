#!/usr/bin/env python
"""
Create offset label points using forced-directed graph drawing
"""
import math

import numpy as np

from .convex_hull import calc_point_offsets as ch_calc_point_offsets

# mass
alpha = 1.0
# column constant
beta = 1.0e-4
# spring constant
k = 0.3
# damping
eta = 0.89
delta_t = 0.5
# tolerance on max kinetic energy before stopping interation
e_tolerance = 1.0e-5


def _norm(
    pts,
    mean=None,
    scaling=None,
):
    """
    Rescale the points so the length-scale it spans fits into the domain while
    having space for the offset points
    """
    x = np.array(pts)
    if mean is None:
        mean = x.mean(axis=0)
    x -= mean

    scaling = 2 * np.max(np.abs(x), axis=0)
    x = x / scaling + 0.5

    return x, mean, scaling


def _denorm(x, mean, scaling):
    return (x - 0.5) * scaling + mean


def _pseudo_coulomb_force(xi, xj, b):
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]
    ds2 = dx * dx + dy * dy
    ds = math.sqrt(ds2)
    ds3 = ds2 * ds
    if ds3 == 0.0:
        const = 0
    else:
        # const = b / (ds2 * ds)
        const = b / (ds * ds)
    return np.array([-const * dx, -const * dy])


def _hooke_force(xi, xj, dij):
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]
    ds = math.sqrt(dx * dx + dy * dy)
    dl = ds - dij
    const = k * dl / ds
    return [const * dx, const * dy]


def calc_offset_points(pts, scale=0.2, callback=None, debug=False):
    N = pts.shape[0]

    if debug:
        from pathlib import Path

        p = Path(__file__)
        np.savetxt(str(p.parent / "points.npz"), pts)

    def update(x, v):
        x_new = np.copy(x)
        v_new = np.copy(v)

        for i in range(m):
            Fx = 0.0
            Fy = 0.0
            for j in range(m):
                if i == j:
                    Fij = _hooke_force(x[i], x_fixed[j], dij)
                else:
                    Fij = _pseudo_coulomb_force(x[i], x[j], b=0.5 * beta)
                    Fij += _pseudo_coulomb_force(x[i], x_fixed[j], b=beta)

                Fx += Fij[0]
                Fy += Fij[1]
            v_new[i][0] = (v_new[i][0] + alpha * Fx * delta_t) * eta
            v_new[i][1] = (v_new[i][1] + alpha * Fy * delta_t) * eta

        for i in range(m):
            x_new[i][0] += v[i][0] * delta_t
            x_new[i][1] += v[i][1] * delta_t

        ekin = np.max(np.sqrt(alpha * (v_new[:, 0] ** 2.0 + v_new[:, 1] ** 2.0))) / N

        return x_new, v_new, ekin

    # pts_offset = ch_calc_point_offsets(pts, scale=scale*4)
    # x, pts_mean, pts_scaling = _norm(pts_offset)
    # x_fixed, _, _ = _norm(pts, mean=pts_mean, scaling=pts_scaling)

    x_fixed, pts_mean, pts_scaling = _norm(pts)
    x = ch_calc_point_offsets(x_fixed, scale=scale)

    dij = scale

    if callback is not None:
        callback(x, x_fixed)

    v = np.zeros_like(x)
    m = len(x)

    e_hist = []
    n = 0
    while True:
        x, v, e_kin = update(x, v)

        n += 1

        e_hist.append(e_kin)

        # using a running mean since there are likely to be oscillations
        e_mean = np.mean(e_hist[-20:])

        if e_mean < e_tolerance:
            break

        # if e_kin > e_old:
        # if e_increasing:
        # pass
        # else:
        # pass
        # else:
        # pass
        # if np.abs(e_kin - e_old) < e_tolerance:
        # break
        # else:
        # e_increasing = False

        # if n % 50 == 0:
        # e_old = e_kin

        if callback is not None:
            callback(x, x_fixed)

    return _denorm(x, mean=pts_mean, scaling=pts_scaling)


def interactive_calc_offset_points(pts, scale=0.2):
    try:
        import tkinter as Tkinter
    except ImportError:
        import Tkinter

    # first call the algorithm to collect the points as they evolve
    results = []

    def callback(x, x_fixed):
        results.append((x, x_fixed))

    calc_offset_points(pts=pts, scale=scale, callback=callback)

    root = Tkinter.Tk()
    canvas = Tkinter.Canvas(root, width=500, height=500, background="yellow")
    canvas.pack()

    ids = []
    lids = []

    results_iter = iter(results)

    UPDATE_FREQ = 10

    def move_oval(id, xi):
        newx = int(xi[0] * 500)
        newy = int(xi[1] * 500)
        canvas.coords(id, newx - 5, newy - 5, newx + 5, newy + 5)

    def move_line(id, xi, xj):
        canvas.coords(
            id, int(xi[0] * 500), int(xi[1] * 500), int(xj[0] * 500), int(xj[1] * 500)
        )

    def update_animation():
        try:
            x, x_fixed = next(results_iter)
        except StopIteration:
            print("done")
            return

        m = len(x)

        if len(ids) == 0:
            for i in range(m):
                xi = x_fixed[i]
                id = canvas.create_oval(245, 245, 255, 255, fill="red")
                move_oval(id, xi)

                xi = x[i]
                id = canvas.create_oval(245, 245, 255, 255, fill="blue")
                ids.append(id)
                move_oval(id, xi)

            for i in range(m):
                id = canvas.create_line(0, 0, 0, 0)
                lids.append(id)
                move_line(id, x_fixed[i], x[i])
        else:
            for i in range(m):
                move_oval(ids[i], xi=x[i])
                move_line(lids[i], x_fixed[i], x[i])

        root.after(UPDATE_FREQ, update_animation)

    root.after(UPDATE_FREQ, update_animation)
    root.mainloop()


if __name__ == "__main__":
    from pathlib import Path

    p = Path(__file__)
    pts = np.loadtxt(p.parent / "points.npz")
    try:
        pass
    except Exception:
        pts = np.array(
            [
                [0.747264, 90.93342],
                [0.806094, 91.614726],
                [0.741519, 88.184657],
                [0.723471, 88.817481],
                [0.777587, 90.1086],
                [0.725995, 87.924487],
                [0.834896, 91.501665],
                [0.722248, 88.508865],
                [0.767785, 88.731781],
            ]
        )

    interactive_calc_offset_points(pts=pts, scale=0.3)
