import satpy
import yaml
import cartopy.crs as ccrs
import xarray as xr
import numpy as np

from . import tiler, Goes16AWS, bbox


def _cleanup_composite_da_attrs(da_composite):
    """
    When we want to save the RGB composite to a netCDF file there are some
    attributes that can't be saved that satpy adds so we remove them here
    """

    def fix_composite_attrs(da):
        fns = [
            ('start_time', lambda v: v.isoformat()),
            ('end_time', lambda v: v.isoformat()),
            ('area', lambda v: None),
            ('prerequisites', lambda v: None),
            ('crs', lambda v: str(v)),
        ]

        for v, fn in fns:
            try:
                da.attrs[v] = fn(da.attrs[v])
            except Exception as e:
                print(e)

        to_delete = [k for (k, v) in da.attrs.items() if v is None]
        for k in to_delete:
            del(da.attrs[k])

    fix_composite_attrs(da_composite)

    return da_composite


def _save_projection_info_to_yaml(channel_fn, fn_meta):
    ds = xr.open_dataset(channel_fn)

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
        projection=serialize_attrs(ds.goes_imager_projection.attrs)
    )

    with open(fn_meta, 'w') as fh:
        yaml.dump(meta_info, stream=fh, default_flow_style=False)


def _create_ccrs_from_yaml(fn_meta):
    with open(fn_meta) as fh:
        meta_info = yaml.load(fh)

    gp = meta_info['projection']

    globe = ccrs.Globe(
        ellipse='sphere', 
        semimajor_axis=gp['semi_major_axis'],
        semiminor_axis=gp['semi_minor_axis']
    )

    return ccrs.Geostationary(
        satellite_height=gp['perspective_point_height'],
        central_longitude=gp['longitude_of_projection_origin'],
        globe=globe
    )


def _load_rgb_files_and_get_composite_da(scene_fns):
    scene = satpy.Scene(reader='abi_l1b', filenames=scene_fns)

    # instruct satpy to load the channels necessary for the `true_color`
    # composite
    scene.load(['true_color'])

    # it is necessary to "resample" here because the different channels are at
    # different spatial resolution. By not passing in an "area" the highest
    # resolution possible will be used
    new_scn = scene.resample(resampler='native')

    # get out a dask-backed DataArray for the composite
    da_truecolor = new_scn['true_color']

    # to be able to crop the DataArray to the bounding box we need to set the
    # projection attribute
    da_truecolor.attrs['crs'] = scene.max_area().to_cartopy_crs()

    return da_truecolor


def _make_composite_filename(scene_fns, bbox_extent):
    key_attrs = Goes16AWS.parse_key(scene_fns[0])
    t_start_str, t_end_str = key_attrs['start_time'], key_attrs['end_time']

    lon_min, lat_min = bbox_extent[0]
    lon_max, lat_max = bbox_extent[1]

    return "rgb_s{}_e{}_{}N_{}W_{}S_{}E.nc".format(
        t_start_str, t_end_str, lat_max, lon_min, lat_min, lon_max
    )


def get_rgb_composite_in_bbox(scene_fns, data_path, bbox_extent,
                              bbox_pad_pct=0.05):
    """
    scene_fns: filenames for the three files containing the channels needed for
    GOES-16 composites

    bbox_extent = (pt_SW, pt_NE)
    """
    fn_nc = _make_composite_filename(scene_fns=scene_fns,
                                     bbox_extent=bbox_extent)

    path_nc = data_path/fn_nc
    path_meta = data_path/fn_nc.replace('.nc', '.meta.yaml')

    bbox_domain = bbox.LatLonBox(bbox_extent)

    if not path_nc.exists():
        da_truecolor = _load_rgb_files_and_get_composite_da(scene_fns=scene_fns)

        da_truecolor_domain = tiler.crop_field_to_latlon_box(
            da=da_truecolor, box=np.array(bbox_domain.get_bounds()).T,
            pad_pct=bbox_pad_pct
        )

        da_truecolor_domain = _cleanup_composite_da_attrs(da_truecolor_domain)

        da_truecolor_domain.to_netcdf(path_nc)

        meta = _save_projection_info_to_yaml(channel_fn=scene_fns[0],
                                             fn_meta=path_meta)

    da = xr.open_dataarray(path_nc)
    da.attrs['crs'] = _create_ccrs_from_yaml(fn_meta=path_meta)

    if not 'source_files' in da.attrs:
        da.attrs['source_files'] = scene_fns

    return da


def rgb_da_to_img(da):
    return satpy.writers.get_enhanced_image(da)
