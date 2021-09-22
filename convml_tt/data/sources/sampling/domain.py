"""
Utilities to create (approximate) square tiles from lat/lon satelite data
"""
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.patches as mpatches
import shapely.geometry as geom
import warnings


class CartesianDomain:
    def __init__(self, l_meridional, l_zonal, x_c=0.0, y_c=0.0):
        self.l_meridional = l_meridional
        self.l_zonal = l_zonal
        self.x_c = x_c
        self.y_c = y_c

    @property
    def spatial_bounds(self):
        """
        The spatial distance bounds of the domain represented by the (x,y)
        position (in meters) of the four corners of the domain
        """
        corners_dir = list(itertools.product([1, -1], [1, -1]))
        corners_dir.insert(0, corners_dir.pop(2))

        corners = np.array([self.l_zonal / 2.0, self.l_meridional / 2.0]) * np.array(
            corners_dir
        )

        corners[..., 0] += self.x_c
        corners[..., 1] += self.y_c

        return corners

    @property
    def spatial_bounds_geometry(self):
        """return a shapely Geometry"""
        return geom.Polygon(self.spatial_bounds)

    def get_grid(self, dx):
        """
        Get an xarray Dataset containing the discrete positions (in meters)
        """
        xmin = self.x_c - self.l_zonal / 2.0 + dx / 2.0
        xmax = self.x_c + self.l_zonal / 2.0 + dx / 2.0
        ymin = self.y_c - self.l_meridional / 2.0 + dx / 2.0
        ymax = self.y_c + self.l_meridional / 2.0 + dx / 2.0
        x_ = np.arange(xmin, xmax, dx)
        y_ = np.arange(ymin, ymax, dx)

        da_x = xr.DataArray(
            x_,
            attrs=dict(long_name="zonal distance", units="m"),
            dims=("x",),
        )
        da_y = xr.DataArray(
            y_,
            attrs=dict(long_name="meridional distance", units="m"),
            dims=("y",),
        )

        ds = xr.Dataset(coords=dict(x=da_x, y=da_y))

        return ds

    def get_grid_extent(self):
        """
        Return grid extent compatible with matplotlib.imshow
        [x0 ,x1, y0, y1] in Cartesian coordinates
        """
        return [
            self.x_c - self.l_zonal / 2.0,
            self.x_c + self.l_zonal / 2.0,
            self.y_c - self.l_meridional / 2.0,
            self.y_c + self.l_meridional / 2.0,
        ]

    def plot_outline(self, ax=None, alpha=0.6, **kwargs):
        if ax is None:
            fig_height = 4
            fig_width = fig_height * self.l_zonal / self.l_meridional
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.margins(0.5)

        bounds_patch = mpatches.Rectangle(
            xy=[self.x_c - self.l_zonal / 2.0, self.y_c - self.l_meridional / 2.0],
            width=self.l_zonal,
            height=self.l_meridional,
            alpha=alpha,
            **kwargs,
        )
        ax.add_patch(bounds_patch)
        return ax

    def serialize(self):
        data = dict(
            x_c=float(self.x_c),
            y_c=float(self.y_c),
            l_zonal=float(self.l_zonal),
            l_meridional=float(self.l_meridional),
        )
        return data

    def validate_dataset(self, ds):
        """
        Ensure the required coordinates exist in the dataset for it to map
        to the defined domain
        """
        # TODO: might want to check for some cartesian coordinates here
        pass


class LocalCartesianDomain(CartesianDomain):
    """
    Domain representing the tangent plane projection centered at specific
    latitude and longitude
    """

    def __init__(
        self,
        central_latitude,
        central_longitude,
        l_meridional,
        l_zonal,
        x_c=0.0,
        y_c=0.0,
    ):
        super().__init__(l_meridional=l_meridional, l_zonal=l_zonal, x_c=x_c, y_c=y_c)
        self.central_latitude = central_latitude
        self.central_longitude = central_longitude

        self.crs = ccrs.LambertAzimuthalEqualArea(
            central_latitude=central_latitude, central_longitude=central_longitude
        )

    @property
    def latlon_bounds(self):
        """
        The spatial distance bounds of the domain represented by the (lat,lon)
        position (in degrees) of the four corners of the domain
        """
        corners = self.spatial_bounds
        latlon_pts = ccrs.PlateCarree().transform_points(
            x=corners[..., 0],
            y=corners[..., 1],
            src_crs=self.crs,
            z=np.zeros_like(corners[..., 0]),
        )

        return latlon_pts

    def get_grid(self, dx):
        """
        Get an xarray Dataset containing the discrete positions (in meters)
        with their lat/lon positions with grid resolution dx (in meters)
        """
        ds_grid_cart = super().get_grid(dx=dx)

        ds_grid = ds_grid_cart.copy()

        ds_grid["x"] = ds_grid.x - self.x_c
        ds_grid["y"] = ds_grid.y - self.y_c

        for c in "xy":
            ds_grid[c].attrs.update(ds_grid_cart[c].attrs)

        x, y = np.meshgrid(ds_grid.x, ds_grid.y, indexing="ij")
        latlon_pts = ccrs.PlateCarree().transform_points(
            x=x, y=y, src_crs=self.crs, z=np.zeros_like(x)
        )

        ds_grid["lon"] = xr.DataArray(
            latlon_pts[..., 0],
            dims=("x", "y"),
            coords=dict(x=ds_grid.x, y=ds_grid.y),
            attrs=dict(standard_name="grid_longitude", units="degree"),
        )
        ds_grid["lat"] = xr.DataArray(
            latlon_pts[..., 1],
            dims=("x", "y"),
            coords=dict(x=ds_grid.x, y=ds_grid.y),
            attrs=dict(standard_name="grid_latitude", units="degree"),
        )

        # the (x,y)-positions are only approximate with the projection
        for c in ["x", "y"]:
            ds_grid[c].attrs["long_name"] = (
                "approximate " + ds_grid[c].attrs["long_name"]
            )

        ds_grid.attrs["crs"] = self.crs

        return ds_grid

    def plot_outline(self, ax=None, alpha=0.6, **kwargs):
        if ax is None:
            fig_height = 4
            fig_width = fig_height * self.l_zonal / self.l_meridional
            fig, ax = plt.subplots(
                figsize=(fig_width, fig_height), subplot_kw=dict(projection=self.crs)
            )
            ax.gridlines(linestyle="--", draw_labels=True)
            ax.coastlines(resolution="10m", color="grey")
            ax.margins(0.5)
        else:
            if getattr(ax, "projection").__class__ != self.crs.__class__:
                warnings.warn(
                    "The outline plot uses a rectangular patch the edges of which"
                    f" are not currently correctly projected unless the {self.crs.__class__.__name__}"
                    " projection is used for the axes"
                )
            pass

        bounds_patch = mpatches.Rectangle(
            xy=[-self.l_zonal / 2.0, -self.l_meridional / 2.0],
            width=self.l_zonal,
            height=self.l_meridional,
            alpha=alpha,
            transform=self.crs,
            **kwargs,
        )
        ax.add_patch(bounds_patch)
        return ax

    def serialize(self):
        data = super().serialize()
        data["central_latitude"] = float(self.central_latitude)
        data["central_longitude"] = float(self.central_longitude)
        return data

    def validate_dataset(self, ds):
        """
        Ensure the required coordinates exist in the dataset for it to map
        to the defined domain
        """
        required_coords = ["lat", "lon"]
        missing_coords = list(filter(lambda c: c not in ds.coords, required_coords))
        if len(missing_coords) > 0:
            raise Exception(
                "The provided dataset is missing the following coordinates "
                f"`{', '.join(missing_coords)}` which are required to make the "
                f" dataset valid for a{self.__class__.__name__} domain"
            )


class LocalCartesianSquareTileDomain(LocalCartesianDomain):
    def __init__(self, central_latitude, central_longitude, size, x_c=0.0, y_c=0.0):
        """
        Create a locally Cartesian square tile with `size` (in meters)
        """
        self.size = size
        super().__init__(
            central_latitude=central_latitude,
            central_longitude=central_longitude,
            l_meridional=size,
            l_zonal=size,
            x_c=x_c,
            y_c=y_c,
        )

    def get_grid(self, N):
        dx = self.size / N
        ds_grid = super().get_grid(dx=dx)
        # the floating point devision in `get_grid` when we're passing in `dx`
        # means that sometimes we might produce an extra set of x- or y-values,
        # so we do an index-selection here to ensure those aren't included
        return ds_grid.isel(x=slice(0, N), y=slice(0, N))


class CartesianSquareTileDomain(CartesianDomain):
    def __init__(self, x_c, y_c, size):
        """
        Create a Cartesian square tile with `size` (in meters) centered at a
        (x,y)=(x_c,y_c) location
        """
        self.size = size
        super().__init__(x_c=x_c, y_c=y_c, l_meridional=size, l_zonal=size)

    def get_grid(self, N):
        dx = self.size / N
        ds_grid = super().get_grid(dx=dx)
        # the floating point devision in `get_grid` when we're passing in `dx`
        # means that sometimes we might produce an extra set of x- or y-values,
        # so we do an index-selection here to ensure those aren't included
        return ds_grid.isel(x=slice(0, N), y=slice(0, N))

    def locate_in_latlon_domain(self, domain):
        """
        Using the (x,y) spatial projection of the local cartesian domain `domain`
        """
        tile_latlon = ccrs.PlateCarree().transform_point(
            x=self.x_c, y=self.y_c, src_crs=domain.crs
        )
        return LocalCartesianSquareTileDomain(
            central_latitude=tile_latlon[1],
            central_longitude=tile_latlon[0],
            size=self.size,
            x_c=self.x_c,
            y_c=self.y_c,
        )


class SourceDataDomain:
    """
    Represents that the domain information should be extracted from a source dataset
    """

    def generate_from_dataset(self, ds):
        """
        Create an actual domain instance from the provided dataset
        """
        if "x" in ds.coords and "y" in ds.coords:
            x_min, x_max = ds.x.min().data, ds.x.max().data
            y_min, y_max = ds.y.min().data, ds.y.max().data

            l_zonal = x_max - x_min
            l_meridinonal = y_max - y_min
            x_c = 0.5 * (x_min + x_max)
            y_c = 0.5 * (y_min + y_max)
            return CartesianDomain(
                l_meridional=l_meridinonal, l_zonal=l_zonal, x_c=x_c, y_c=y_c
            )
        elif "lat" in ds.coords and "lon" in ds.coords:
            raise NotImplementedError(LocalCartesianDomain.__name__)
        else:
            raise NotImplementedError(ds.coords)


def deserialise_domain(data):
    if "central_longitude" in data and "central_latitude" in data:
        return LocalCartesianDomain(**data)
    else:
        return CartesianDomain(**data)
