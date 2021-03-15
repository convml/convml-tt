"""
Functions to create RGB true-color composites from GOES satellite radiance
measurements
"""
from pathlib import Path
import cartopy.crs as ccrs
import numpy as np
import satpy
import satpy.composites.abi
import satpy.composites.cloud_products
import satpy.composites.viirs
import satpy.enhancements
import xarray as xr
import yaml
from satdata import Goes16AWS
import luigi

from . import bbox, tiler
from ...datasource import GenericDatasource


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


def _load_rgb_files_and_get_composite_da(scene_fns):
    """
    Use satpy to generate a xr.DataArray with a true-colour RGB image data for
    the scene filenames in `scene_fns` (which are assumed to contain the first
    three radiance channels)
    """
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


def rgb_da_to_img(da):
    # need to sort by y otherize resulting image is flipped... there must be a
    # better way
    da_ = da.sortby("y", ascending=False)
    return satpy.writers.get_enhanced_image(da_)


class RGBCompositeNetCDFFile(luigi.LocalTarget):
    """
    Special target which together with the xarray.DataArray containing the RGB
    image values in an xr.DataArray also stores the projection info about the
    underlying data
    """
    def save(self, da_truecolor, source_fns):
        Path(self.fn).parent.mkdir(exist_ok=True, parents=True)
        self._save_scene_meta(source_fns=source_fns, fn_meta=self.path_meta)
        da_truecolor.to_netcdf(self.fn)

    @property
    def path_meta(self):
        return self.fn.replace(".nc", ".meta.yaml")

    def open(self):
        try:
            da = xr.open_dataarray(self.fn)
        except Exception:
            print("Error opening `{}`".format(self.fn))
            raise
        meta_info = self._load_scene_meta(fn_meta=self.path_meta)
        da.attrs.update(meta_info)

        return da

    @staticmethod
    def _save_scene_meta(source_fns, fn_meta):
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
                    except ValueError as ex:
                        raise NotImplementedError(
                            "not sure how to handle `{}`:{}".format(k, v)
                        ) from ex
            return new_attrs

        meta_info = dict(
            projection=serialize_attrs(ds.goes_imager_projection.attrs),
            source_files=[str(p) for p in source_fns],
        )

        with open(fn_meta, "w") as fh:
            yaml.dump(meta_info, stream=fh, default_flow_style=False)

    @staticmethod
    def _load_scene_meta(fn_meta):
        with open(fn_meta) as fh:
            meta_info = yaml.load(fh)

        proj_info = meta_info["projection"]

        globe = ccrs.Globe(
            ellipse="sphere",
            semimajor_axis=proj_info["semi_major_axis"],
            semiminor_axis=proj_info["semi_minor_axis"],
        )

        crs = ccrs.Geostationary(
            satellite_height=proj_info["perspective_point_height"],
            central_longitude=proj_info["longitude_of_projection_origin"],
            globe=globe,
        )

        meta_info["crs"] = crs
        del meta_info["projection"]
        return meta_info


class CreateRGBScene(luigi.Task):
    """
    Create RGB composite scene from GOES-16 radiance files
    """

    scene_id = luigi.Parameter()
    dataset_path = luigi.Parameter()

    def _load_datasource_info(self):
        return GenericDatasource.load(self.dataset_path)

    def requires(self):
        return d.fetch_source_data()

    def run(self):
        d = GenericDatasource.load(self.dataset_path)

        all_source_data = self.input().read()
        if self.scene_id not in all_source_data:
            raise Exception(
                "scene `{}` is missing from the source data file"
                "".format(self.scene_id)
            )
        else:
            scene_fns = [
                Path(self.dataset_path) / SOURCE_DIR / p
                for p in all_source_data[self.scene_id]
            ]
        # OBS: should probably check that the channels necessary for RGB scene
        # generation are the ones loaded here

        da_truecolor = _load_rgb_files_and_get_composite_da(
            scene_fns=scene_fns
        )

        bbox_domain = d.get_domain(da_scene=da_truecolor)
        domain_bbox_pad_frac = getattr(d, "domain_bbox_pad_frac", 0.1)

        da_truecolor_domain = tiler.crop_field_to_latlon_box(
            da=da_truecolor,
            box=np.array(bbox_domain.get_bounds()).T,
            pad_pct=domain_bbox_pad_frac,
        )

        da_truecolor_domain = _cleanup_composite_da_attrs(da_truecolor_domain)

        self.output().save(da_truecolor=da_truecolor_domain, source_fns=scene_fns)

    def output(self):
        fn = "{}.nc".format(self.scene_id)
        p = Path(self.dataset_path) / "composites" / "original_cropped" / fn
        t = RGBCompositeNetCDFFile(str(p))
        return t
