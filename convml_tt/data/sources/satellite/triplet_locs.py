"""
Utility for producing tile sampling within (lat, lon) bounding box
"""
import numpy as np
from scipy.constants import pi


def _est_tile_size_deg(loc, tile_size):
    _, lat0 = loc
    r_earth = 6371e3  # Earth's radius in m
    tile_size_deg = np.rad2deg(tile_size / (r_earth * np.cos(np.deg2rad(lat0))))
    return tile_size_deg


def _point_valid(lon, lat, tiling_bbox, tile_size):
    h_ts = 0.5 * _est_tile_size_deg(loc=(lon, lat), tile_size=tile_size)

    (lon_min, lat_min), (lon_max, lat_max) = tiling_bbox
    within_lon_range = lon_min + h_ts <= lon <= lon_max - h_ts
    within_lat_range = lat_min + h_ts <= lat <= lat_max - h_ts
    return within_lon_range and within_lat_range


def _generate_latlon_within_bbox(tiling_bbox, tile_size):
    (lon_min, lat_min), (lon_max, lat_max) = tiling_bbox

    lat = lat_min + (lat_max - lat_min) * np.random.random()
    lon = lon_min + (lon_max - lon_min) * np.random.random()

    if not _point_valid(lon, lat, tile_size=tile_size, tiling_bbox=tiling_bbox):
        return _generate_latlon_within_bbox(
            tile_size=tile_size, tiling_bbox=tiling_bbox
        )
    else:
        return (lon, lat)


def _perturb_loc(loc, scaling, tile_size, tiling_bbox):
    theta = 2 * pi * np.random.random()

    tile_size_deg = _est_tile_size_deg(loc=loc, tile_size=tile_size)
    r_offset = scaling * tile_size_deg * np.random.normal(loc=1.0, scale=0.1)

    dlon = r_offset * np.cos(theta)
    dlat = r_offset * np.sin(theta)

    new_loc = (loc[0] + dlon, loc[1] + dlat)
    if _point_valid(
        lon=new_loc[0], lat=new_loc[1], tile_size=tile_size, tiling_bbox=tiling_bbox
    ):
        return new_loc
    else:
        return _perturb_loc(
            loc=loc, scaling=scaling, tile_size=tile_size, tiling_bbox=tiling_bbox
        )


def generate_triplet_locs(tile_size, tiling_bbox, neigh_dist_scaling=0.1):
    """
    Generate locations for a triplet (anchor, neighbour and distance) with
    `tile_size` (in meters) inside `tiling_bbox` given as [(lon_min, lat_min),
    (lon_max, lat_max)]
    """

    anchor_loc = _generate_latlon_within_bbox(
        tiling_bbox=tiling_bbox, tile_size=tile_size
    )
    neighbor_loc = _perturb_loc(
        anchor_loc,
        scaling=neigh_dist_scaling,
        tiling_bbox=tiling_bbox,
        tile_size=tile_size,
    )
    dist_loc = _generate_latlon_within_bbox(
        tiling_bbox=tiling_bbox, tile_size=tile_size
    )

    return [anchor_loc, neighbor_loc, dist_loc]
