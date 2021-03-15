from pathlib import Path
import warnings
import xesmf
import os
import cartopy.crs as ccrs
import xarray as xr
import numpy as np

from .grid import make_rect_latlon_grid


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


def _create_tile_regridder(weights_path=Path("/tmp/regridding")):
    self.regridder_tmpdir = Path(tempfile.mkdtemp(dir=weights_path))


def rect_resample(
    da,
    lat0,
    lon0,
    size,
    method="bilinear",
    regridder_weights_dir=Path("/tmp/regridding"),
    keep_attrs=True,
):
    """
    Resample `da` onto a locally rectangular grid with `size` and centered on
    `lat0`, `lon0`
    """
    if not ("lat" in da.coords and "lon" in da.coords):
        raise NotImplementedError(da.coords)

    new_grid = make_rect_latlon_grid(size=size, lat0=lat0, lon0=lon0)
    old_grid = xr.Dataset(coords=da.coords)

    if not hasattr(da, "crs"):
        raise Exception(
            "The provided DataArray doesn't have a "
            "projection provided. Please set the `crs` "
            "attribute to contain a cartopy projection"
        )

    # ensure we have somewhere to store the griddings weights
    regridder_weights_dir.mkdir(exist_ok=True, parents=True)

    # compute the latlon coordinates of the dataset we're regridding
    # XXX: aren't these stored with the file?
    latlon_old = ccrs.PlateCarree().transform_points(
        da.crs,
        *np.meshgrid(da.x.values, da.y.values),
    )[:, :, :2]

    old_grid["lat"] = (("y", "x"), latlon_old[..., 1])
    old_grid["lon"] = (("y", "x"), latlon_old[..., 0])

    nx_in, ny_in = da.x.shape[0], da.y.shape[0]
    nx_out, ny_out = int(new_grid.x.count()), int(new_grid.y.count())

    regridder_weights_fn = (
        "{method}_{ny_in}x{nx_in}_{ny_out}x{nx_out}"
        "__{lat0}_{lon0}.nc".format(
            lon0=lon0,
            lat0=lat0,
            method=method,
            ny_in=ny_in,
            nx_in=nx_in,
            nx_out=nx_out,
            ny_out=ny_out,
        )
    )

    regridder_weights_fn = str(regridder_weights_dir / regridder_weights_fn)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            regridder = SilentRegridder(
                filename=regridder_weights_fn,
                reuse_weights=True,
                ds_in=old_grid,
                ds_out=new_grid,
                method=method,
            )
        except ValueError as ex:
            raise Exception("something went wrong" " during regridding :(") from ex

    da_resampled = regridder(da)

    da_resampled["x"] = new_grid.x
    da_resampled["y"] = new_grid.y

    if keep_attrs:
        da_resampled.attrs.update(da.attrs)

    return da_resampled


def get_pyresample_area_def(N, L, lat0, lon0):
    """
    When using satpy scenes we're better off using pyresample instead of
    xesmf since it appears faster (I think because it uses dask)
    """
    from pyresample import geometry

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
        "lon_0": lon0,
        "lat_0": lat0,
    }
    area_def = geometry.AreaDefinition(
        area_id, description, proj_id, proj_dict, x_size, y_size, area_extent
    )

    return area_def
