import cartopy.crs as ccrs
import xesmf
import xarray as xr
import numpy as np

import warnings

import os

from xesmf.backend import esmf_regrid_build, esmf_regrid_finalize
from pathlib import Path
from ..sampling.domain import LocalCartesianDomain


class SilentRegridder(xesmf.Regridder):
    def _write_weight_file(self):
        if os.path.exists(self.filename):
            if self.reuse_weights:
                return  # do not compute it again, just read it
            else:
                os.remove(self.filename)

        regrid = esmf_regrid_build(
            self._grid_in, self._grid_out, self.method, filename=self.filename
        )
        esmf_regrid_finalize(regrid)  # only need weights, not regrid object


def resample(
    domain,
    da,
    dx,
    method="bilinear",
    crop_pad_pct=0.1,
    keep_attrs=False,
    regridder_tmpdir=Path("/tmp/regridding"),
):
    """
    Resample a xarray DataArray onto this tile with grid made of NxN points
    """
    old_grid = xr.Dataset(coords=da.coords)

    if isinstance(domain, LocalCartesianDomain):
        if not hasattr(da, "crs"):
            raise Exception(
                "The provided DataArray doesn't have a "
                "projection provided. Please set the `crs` "
                "attribute to contain a cartopy projection"
            )

        latlon_old = ccrs.PlateCarree().transform_points(
            da.crs,
            *np.meshgrid(da.x.values, da.y.values),
        )[:, :, :2]

        old_grid["lat"] = (("y", "x"), latlon_old[..., 1])
        old_grid["lon"] = (("y", "x"), latlon_old[..., 0])

    new_grid = domain.get_grid(dx=dx)

    Nx_in, Ny_in = da.x.shape[0], da.y.shape[0]
    Nx_out, Ny_out = int(new_grid.x.count()), int(new_grid.y.count())

    regridder_weights_fn = "{method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}" ".nc".format(
        method=method,
        Ny_in=Ny_in,
        Nx_in=Nx_in,
        Nx_out=Nx_out,
        Ny_out=Ny_out,
    )

    regridder_weights_fn = str(regridder_tmpdir / regridder_weights_fn)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regridder = SilentRegridder(
            filename=regridder_weights_fn,
            reuse_weights=True,
            ds_in=old_grid,
            ds_out=new_grid,
            method=method,
        )

    da_resampled = regridder(da)

    da_resampled["x"] = new_grid.x
    da_resampled["y"] = new_grid.y

    if keep_attrs:
        da_resampled.attrs.update(da.attrs)

    return da_resampled


def get_pyresample_area_def(N, domain):
    """
    When using satpy scenes we're better off using pyresample instead of
    xesmf since it appears faster (I think because it uses dask)
    """
    from pyresample import geometry

    L = domain.size
    area_id = "tile"
    description = "Tile local cartesian grid"
    proj_id = "ease_tile"
    x_size = N
    y_size = N
    area_extent = (-L, -L, L, L)
    proj_dict = {
        "a": 6371228.0,
        "units": "m",
        "proj": "laea",
        "lon_0": domain.lon0,
        "lat_0": domain.lat0,
    }
    area_def = geometry.AreaDefinition(
        area_id, description, proj_id, proj_dict, x_size, y_size, area_extent
    )

    return area_def
