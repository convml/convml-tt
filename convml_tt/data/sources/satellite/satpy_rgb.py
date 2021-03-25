import satpy
import yaml
import cartopy.crs as ccrs
import xarray as xr
import numpy as np

from satdata import Goes16AWS
from . import tiler, bbox

import satpy.composites.viirs
import satpy.composites.abi
import satpy.composites.cloud_products
import satpy.enhancements


def _cleanup_composite_da_attrs(da_composite):
    """
    When we want to save the RGB composite to a netCDF file there are some
    attributes that can't be saved that satpy adds so we remove them here
    """

    def fix_composite_attrs(da):
        fns = [
            ("start_time", lambda v: v.isoformat()),
            ("end_time", lambda v: v.isoformat()),
            ("area", lambda v: None),
            ("prerequisites", lambda v: None),
            ("crs", lambda v: str(v)),
            ("orbital_parameters", lambda v: None),
        ]

        for v, fn in fns:
            try:
                da.attrs[v] = fn(da.attrs[v])
            except Exception as e:
                print(e)

        to_delete = [k for (k, v) in da.attrs.items() if v is None]
        for k in to_delete:
            del da.attrs[k]

    fix_composite_attrs(da_composite)

    return da_composite


def save_scene_meta(source_fns, fn_meta):
    # get projection info just from the first source file
    ds = xr.open_dataset(source_fns[0])

    def serialize_attrs(attrs):
        new_attrs = {}
        for k, v in attrs.items():
            if isinstance(v, str):
                new_attrs[k] = v
            else:
                try:
                    new_attrs[k] = float(v)
                except:
                    raise NotImplementedError(
                        "not sure how to handle `{}`:{}".format(k, v)
                    )
        return new_attrs

    meta_info = dict(
        projection=serialize_attrs(ds.goes_imager_projection.attrs),
        source_files=[str(p) for p in source_fns],
    )

    with open(fn_meta, "w") as fh:
        yaml.dump(meta_info, stream=fh, default_flow_style=False)


def load_scene_meta(fn_meta):
    with open(fn_meta) as fh:
        meta_info = yaml.load(fh)

    gp = meta_info["projection"]

    globe = ccrs.Globe(
        ellipse="sphere",
        semimajor_axis=gp["semi_major_axis"],
        semiminor_axis=gp["semi_minor_axis"],
    )

    crs = ccrs.Geostationary(
        satellite_height=gp["perspective_point_height"],
        central_longitude=gp["longitude_of_projection_origin"],
        globe=globe,
    )

    meta_info["crs"] = crs
    del meta_info["projection"]
    return meta_info


def load_rgb_files_and_get_composite_da(scene_fns):
    scene = satpy.Scene(reader="abi_l1b", filenames=scene_fns)

    # instruct satpy to load the channels necessary for the `true_color`
    # composite
    scene.load(["true_color"])

    # it is necessary to "resample" here because the different channels are at
    # different spatial resolution. By not passing in an "area" the highest
    # resolution possible will be used
    new_scn = scene.resample(resampler="native")

    # get out a dask-backed DataArray for the composite
    da_truecolor = new_scn["true_color"]

    if "crs" in da_truecolor:
        da_truecolor = da_truecolor.drop("crs")

    # to be able to crop the DataArray to the bounding box we need to set the
    # projection attribute
    da_truecolor.attrs["crs"] = scene.max_area().to_cartopy_crs()

    return da_truecolor


def make_composite_filename(scene_fns, bbox_extent):
    key_attrs = Goes16AWS.parse_key(scene_fns[0])
    t_start_str, t_end_str = key_attrs["start_time"], key_attrs["end_time"]

    lon_min, lat_min = bbox_extent[0]
    lon_max, lat_max = bbox_extent[1]

    return "rgb_s{}_e{}_{}N_{}W_{}S_{}E.nc".format(
        t_start_str, t_end_str, lat_max, lon_min, lat_min, lon_max
    )


def get_rgb_composite_in_bbox(scene_fns, data_path, bbox_extent, bbox_pad_pct=0.05):
    """
    scene_fns: filenames for the three files containing the channels needed for
    GOES-16 composites

    bbox_extent = (pt_SW, pt_NE)
    """
    fn_nc = make_composite_filename(scene_fns=scene_fns, bbox_extent=bbox_extent)

    path_nc = data_path / fn_nc
    path_meta = data_path / fn_nc.replace(".nc", ".meta.yaml")

    data_path.mkdir(exist_ok=True, parents=True)

    bbox_domain = bbox.LatLonBox(bbox_extent)

    if not path_nc.exists():
        da_truecolor = load_rgb_files_and_get_composite_da(scene_fns=scene_fns)

        da_truecolor_domain = tiler.crop_field_to_latlon_box(
            da=da_truecolor,
            box=np.array(bbox_domain.get_bounds()).T,
            pad_pct=bbox_pad_pct,
        )

        da_truecolor_domain = _cleanup_composite_da_attrs(da_truecolor_domain)

        da_truecolor_domain.to_netcdf(path_nc)

        meta = _save_scene_meta(source_fns=scene_fns, fn_meta=path_meta)

    return da


def rgb_da_to_img(da):
    # need to sort by y otherize resulting image is flipped... there must be a
    # better way
    da_ = da.sortby("y", ascending=False)
    return satpy.writers.get_enhanced_image(da_)
