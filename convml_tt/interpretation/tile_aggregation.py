"""
Selection of routines to perform aggregation calculations over a set of tiles,
for example calculating the mean Radiance for a given channel and tile

For this we need to load the tile definitions, identify the data files required
to aggregate over and do the aggregation operation
"""
import datetime

import yaml
from tqdm import tqdm
import numpy as np
import xarray as xr
from scipy.constants import pi
from pathlib import Path
import satdata

from ..architectures.triplet_trainer import TileType

from PIL import Image as Image_PIL
from fastai.vision.image import Image as Image_fastai, image2np

import convorg


def _get_storage_keys_for_channel(tiles, channel, cli, dt_max):
    """work out the storage keys that we need get the requested channel for
    each tile"""

    def find_key(tile):
        fh_source_ch1 = tile.meta["rgb_source_files"][0]
        key_parts = satdata.Goes16AWS.parse_key(fh_source_ch1, parse_times=True)

        t_start = key_parts["start_time"]
        t_end = key_parts["end_time"]

        t = t_start

        keys = cli.query(time=t, dt_max=dt_max, region="F", channel=channel)
        key = keys[0]

        return key

    return [find_key(tile) for tile in tqdm(tiles)]


def _fetch_channel_radiance(key, cli):
    fn = cli.download(key)[0]

    if fn is None:
        raise Exception("cannot load {}".format(fn))

    ds = xr.open_dataset(fn)
    ds = satdata.processing.set_projection_attribute_and_scale_coords(ds)

    da = ds.Rad
    da.attrs["crs"] = ds.crs

    da.attrs["aws_s3_key"] = key

    da = da.drop(["t", "x_image", "y_image"])

    return da


def _perform_tile_reduction(tile, da_channel, op, resample):
    if resample:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da_reduc = op(tile.resample(da_channel, N=N))
    else:
        # use less accurate reduction where bounding box on tile is
        # cropped in coordinate system of input
        da_reduc = op(tile.crop_field(da=da_channel, pad_pct=0.0))

    da_reduc.attrs.update(da_channel.attrs)
    del da_reduc.attrs["crs"]
    da_reduc.attrs["tile"] = tile
    da_reduc["tile_id"] = int(tile.tile_id)
    return da_reduc


def aggregate_channel_over_tiles(
    tiles, channel, op, resample, cli, dt_max=datetime.timedelta(minutes=10)
):

    storage_keys = _get_storage_keys_for_channel(
        tiles=tiles, channel=channel, cli=cli, dt_max=dt_max
    )

    unique_keys = list(set(storage_keys))
    channels_data = dict(
        [(key, _fetch_channel_radiance(key, cli)) for key in tqdm(unique_keys)]
    )

    channel_data_for_each_tile = [channels_data[key] for key in storage_keys]

    da_channel = xr.concat(
        [
            _perform_tile_reduction(
                tile, da_channel=da_channel, resample=resample, op=op
            )
            for (tile, da_channel) in tqdm(list(zip(tiles, channel_data_for_each_tile)))
        ],
        dim="tile_id",
    )

    da_channel.name = "{}_{}".format(da_channel.name, op.__name__)
    da_channel.attrs["resampled"] = str(resample)
    da_channel.attrs["long_name"] = "{} {}".format(da_channel.long_name, op.__name__)
    da_channel["channel"] = channel

    # give this `source_path` attribute so we can plot examples with the
    # annotated scatterplot
    da_channel.attrs["source_path"] = str(tiles[0].data_path)

    return da_channel


def scale_to_approximate_flux(da_rad):
    channel = int(da_rad.channel)
    # get channel width in meters
    df_channel = satdata.Goes16AWS.CHANNEL_WIDTHS[channel] * 1.0e6

    assert da_rad.units == "W m-2 sr-1 um-1"

    da_flux_approx = 4 * pi * channel * da_rad

    da_flux_approx.attrs["units"] = "W/m^2"
    da_flux_approx.attrs["long_name"] = "approximate channel flux"

    da_flux_approx.name = "F_approx"

    return da_flux_approx


def calc_conv_org_metrics(img, greyscale_threshold=120.0, plot_mask=False):
    if isinstance(img, Image_fastai):
        # https://stackoverflow.com/a/12201744
        img_data = np.transpose(img.data, (2, 1, 0))
        da_img_gs = xr.DataArray(np.dot(img_data, [0.2989, 0.5870, 0.1140])) * 255.0
    else:
        da_img_gs = xr.DataArray(np.array(img.convert("LA"))[..., 0])
    da_mask = da_img_gs > greyscale_threshold

    if plot_mask:
        da_mask.plot()

    try:
        iorg_val = convorg.iorg(da_mask.values)
    except:
        iorg_val = np.nan

    try:
        scai_val = convorg.scai(da_mask.values)
    except:
        scai_val = np.nan

    try:
        hausdorff_val = convorg.hausdorff_dimension(da_mask.values)
    except:
        hausdorff_val = np.nan

    ds = xr.merge(
        [
            xr.DataArray(iorg_val, name="iorg"),
            xr.DataArray(scai_val, name="scai"),
            xr.DataArray(hausdorff_val, name="hausdorff_dimension"),
        ]
    )
    ds.attrs["greyscale_threshold"] = greyscale_threshold
    return ds


def aggregate_fn_over_tiles(
    triplets, method, tile_type=TileType.ANCHOR, method_kwargs={}
):
    fn_aggregated = "tiles_agg__{}.nc".format(method)

    if not Path(fn_aggregated).exists():
        fn = globals()["calc_" + method]
        ds = xr.concat(
            [
                fn(triplet[tile_type.value], **method_kwargs)
                for triplet in tqdm(triplets)
            ],
            dim="tile_id",
        )
        ds.to_netcdf(fn_aggregated)
    else:
        ds = xr.open_dataset(fn_aggregated)
    return ds
