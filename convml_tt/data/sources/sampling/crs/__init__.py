import numpy as np
import xarray as xr
import logging

from .mapping import CFProjection

log = logging.getLogger()


def parse_cf(ds_or_da, varname=None):
    """Parse dataset or dataarray for coordinate system metadata according to CF conventions.

    Interpret the grid mapping metadata in the dataset according to the Climate and
    Forecasting (CF) conventions (see `Appendix F: Grid Mappings
    <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#appendix-grid-mappings>`_
    ).

    This method operates on individual data variables within the dataset, so do not be
    suprised if information not associated with individual data variables is not
    preserved.

    Parameters
    ----------
    varname : str or iterable of str, optional
        Name of the variable(s) to extract from the dataset while parsing for CF metadata.
        Defaults to all variables.

    Returns
    -------
    `cartopy.crs.Projection`
        Parsed cartopy projection

    """
    if isinstance(ds_or_da, xr.Dataset):
        ds = ds_or_da
        if varname is None:
            # If no varname is given, parse all variables in the dataset
            varname = list(ds.data_vars)

        if np.iterable(varname) and not isinstance(varname, str):
            # If non-string iterable is given, apply recursively across the varnames
            subset = xr.merge(
                [parse_cf(ds, single_varname) for single_varname in varname]
            )
            subset.attrs = ds.attrs
            return subset

        da = ds[varname]
    elif isinstance(ds_or_da, xr.DataArray):
        da = ds_or_da
        if varname is not None:
            raise Exception("`varname` should only be provided for datasets")
    else:
        raise NotImplementedError(type(ds_or_da))

    # Attempt to build the crs coordinate
    crs = None
    if "grid_mapping" in da.attrs:
        # Use given CF grid_mapping
        proj_name = da.attrs["grid_mapping"]

        if isinstance(ds_or_da, xr.Dataset) and proj_name in ds_or_da.variables:
            proj_var = ds_or_da.variables[proj_name]
        elif isinstance(ds_or_da, xr.DataArray) and proj_name in ds_or_da.coords:
            proj_var = ds_or_da.coords[proj_name]
        else:
            log.warning(
                "Could not find variable corresponding to the value of "
                f"grid_mapping: {proj_name}"
            )

        crs = CFProjection(proj_var.attrs)

    if crs is None:
        raise Exception("Didn't find any valid projection information")

    return crs.to_cartopy()
