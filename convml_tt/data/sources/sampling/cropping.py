import numpy as np
import cartopy.crs as ccrs

from ..sampling.crs import parse_cf as parse_cf_crs


def crop_field_to_bbox(da, x_range, y_range, pad_pct=0.1, x_dim="x", y_dim="y"):
    x_min, x_max = x_range
    y_min, y_max = y_range

    lx = x_max - x_min
    ly = y_max - y_min

    x_min -= pad_pct * lx
    y_min -= pad_pct * ly
    x_max += pad_pct * lx
    y_max += pad_pct * ly

    if da[x_dim][0] > da[x_dim][-1]:
        x_slice = slice(x_max, x_min)
    else:
        x_slice = slice(x_min, x_max)

    if da[y_dim][0] > da[y_dim][-1]:
        y_slice = slice(y_max, y_min)
    else:
        y_slice = slice(y_min, y_max)

    return da.sel({x_dim: x_slice, y_dim: y_slice})


class DomainBoundsOutsideOfInputException(Exception):
    pass


def _has_spatial_coord(da, c):
    return c in da and da[c].attrs.get("units") == "m"


def crop_field_to_domain(domain, da, pad_pct=0.1):
    if _has_spatial_coord(da=da, c="x") and _has_spatial_coord(da=da, c="y"):
        raise NotImplementedError
    elif "lat" in da.coords and "lon" in da.coords:
        x_dim, y_dim = "lon", "lat"
        xs = domain.latlon_bounds[..., 0]
        ys = domain.latlon_bounds[..., 1]
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        if da[x_dim][-1] > 180.0:
            if x_max < 0.0:
                x_min += 360.0
                x_max += 360.0
            else:
                raise NotImplementedError
        x_range = [x_min, x_max]
        y_range = [y_min, y_max]
    elif "grid_mapping" in da.attrs:
        x_dim, y_dim = "x", "y"
        crs = parse_cf_crs(da)
        # the source data is stored in its own projection and so we want to
        # crop using the coordinates of this projection
        latlon_box = domain.latlon_bounds
        xs, ys, _ = crs.transform_points(ccrs.PlateCarree(), *latlon_box.T).T
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        x_range = [x_min, x_max]
        y_range = [y_min, y_max]
    else:
        raise NotImplementedError(da)

    da_cropped = crop_field_to_bbox(
        da=da,
        x_range=x_range,
        y_range=y_range,
        pad_pct=pad_pct,
        x_dim=x_dim,
        y_dim=y_dim,
    )

    if da_cropped[x_dim].count() == 0 or da_cropped[y_dim].count() == 0:
        raise DomainBoundsOutsideOfInputException

    return da_cropped
