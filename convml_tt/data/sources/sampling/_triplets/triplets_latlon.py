from scipy.constants import pi
import numpy as np


from .latlon import Tile

try:
    from . import satpy_rgb

    HAS_SATPY = True
except ImportError:
    HAS_SATPY = False


def triplet_generator(
    da_target_scene,
    tile_size,
    tiling_bbox,
    tile_N,
    da_distant_scene=None,
    neigh_dist_scaling=1.0,
    distant_dist_scaling=10.0,
):
    # generate (lat, lon) locations inside tiling_box

    def _est_tile_size_deg(loc):
        _, lat0 = loc
        R = 6371e3  # Earth's radius in m
        tile_size_deg = np.rad2deg(tile_size / (R * np.cos(np.deg2rad(lat0))))
        return tile_size_deg

    def _point_valid(lon, lat):
        h_ts = 0.5 * _est_tile_size_deg(loc=(lon, lat))

        (lon_min, lat_min), (lon_max, lat_max) = tiling_bbox
        try:
            assert lon_min + h_ts <= lon <= lon_max - h_ts
            assert lat_min + h_ts <= lat <= lat_max - h_ts
            return True
        except ValueError:
            return False

    def _generate_latlon():
        (lon_min, lat_min), (lon_max, lat_max) = tiling_bbox

        lat = lat_min + (lat_max - lat_min) * np.random.random()
        lon = lon_min + (lon_max - lon_min) * np.random.random()

        if not _point_valid(lon, lat):
            return _generate_latlon()
        else:
            return (lon, lat)

    def _perturb_loc(loc, scaling):
        theta = 2 * pi * np.random.random()

        tile_size_deg = _est_tile_size_deg(loc=loc)

        r = scaling * tile_size_deg * np.random.normal(loc=1.0, scale=0.1)

        dlon = r * np.cos(theta)
        dlat = r * np.sin(theta)

        new_loc = (loc[0] + dlon, loc[1] + dlat)
        if _point_valid(lon=new_loc[0], lat=new_loc[1]):
            return new_loc
        else:
            return _perturb_loc(loc=loc, scaling=scaling)

    anchor_loc = _generate_latlon()
    neighbor_loc = _perturb_loc(anchor_loc, scaling=neigh_dist_scaling)

    if da_distant_scene is None:
        while True:
            dist_loc = _perturb_loc(anchor_loc, scaling=distant_dist_scaling)
            if _point_valid(dist_loc):
                break
    else:
        dist_loc = _generate_latlon()

    locs = [anchor_loc, neighbor_loc, dist_loc]

    tiles = [Tile(lat0=lat, lon0=lon, size=tile_size) for (lon, lat) in locs]

    # create a list of the three scenes used for creating triplets
    da_scene_set = [da_target_scene, da_target_scene]
    if da_distant_scene is None:
        da_scene_set.append(da_target_scene)
    else:
        da_scene_set.append(da_distant_scene)

    # on each of the three scenes use the three tiles to create a resampled
    # image
    try:
        return [
            (tile, tile.create_true_color_img(da_scene, resampling_N=tile_N))
            for (tile, da_scene) in zip(tiles, da_scene_set)
        ]
    except Tile.TileBoundsOutsideOfInputException:
        return triplet_generator(
            da_target_scene,
            tile_size,
            tiling_bbox,
            tile_N,
            da_distant_scene,
            neigh_dist_scaling,
            distant_dist_scaling,
        )


def create_true_color_img(tile, da_scene, resampling_dx):
    if isinstance(da_scene, list):
        das_channels_resampled = [
            tile.resample(da, dx=resampling_dx) for da in da_scene
        ]
        return create_true_color_img(das_channels=das_channels_resampled)
    else:
        if not HAS_SATPY:
            raise Exception(
                "Must have satpy installed to be able to "
                "RGB composites with satpy"
            )

        da_tile_rgb = tile.resample(da=da_scene, dx=resampling_dx, keep_attrs=True)

        return satpy_rgb.rgb_da_to_img(da_tile_rgb)
