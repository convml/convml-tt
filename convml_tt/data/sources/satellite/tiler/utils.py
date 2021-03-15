import numpy as np


def crop_field_to_latlon_box(da, latlon_pts, pad_pct=0.1):
    """
    Crop xr.Dataset which as variables for `lat` and `lon` by bounding-box
    containing points in `latlon_pts`
    """
    xs, ys, _ = da.crs.transform_points(ccrs.PlateCarree(), *latlon_pts).T

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    lx = x_max - x_min
    ly = y_max - y_min

    x_min -= pad_pct * lx
    y_min -= pad_pct * ly
    x_max += pad_pct * lx
    y_max += pad_pct * ly

    if da.x[0] > da.x[-1]:
        x_slice = slice(x_max, x_min)
    else:
        x_slice = slice(x_min, x_max)

    if da.y[0] > da.y[-1]:
        y_slice = slice(y_max, y_min)
    else:
        y_slice = slice(y_min, y_max)

    return da.sel(x=x_slice, y=y_slice)
