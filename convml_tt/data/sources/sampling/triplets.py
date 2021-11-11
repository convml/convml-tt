import numpy as np
from regridcart import LocalCartesianDomain
from scipy.constants import pi

from ..sampling import CartesianSquareTileDomain


def generate_randomly_located_tile(domain, tile_size):
    """
    Generate a tile domain of a specific `tile_siez` that fits inside `domain`
    """
    domain_bounds = domain.spatial_bounds
    domain_bounds_geometry = domain.spatial_bounds_geometry

    d_xmin = np.min(domain_bounds[..., 0]) + tile_size / 2.0
    d_xmax = np.max(domain_bounds[..., 0]) - tile_size / 2.0
    d_ymin = np.min(domain_bounds[..., 1]) + tile_size / 2.0
    d_ymax = np.max(domain_bounds[..., 1]) - tile_size / 2.0

    x_t = d_xmin + (d_xmax - d_xmin) * np.random.random()
    y_t = d_ymin + (d_ymax - d_ymin) * np.random.random()

    tile_domain = CartesianSquareTileDomain(x_c=x_t, y_c=y_t, size=tile_size)
    if isinstance(domain, LocalCartesianDomain):
        tile_domain = tile_domain.locate_in_latlon_domain(domain=domain)

    domain_area = domain.spatial_bounds_geometry.area
    tile_area = tile_domain.spatial_bounds_geometry.area

    if not domain_area > tile_area:
        raise Exception(
            "Your sampling domain is too small to fit the tile size you have requested "
            f"(domain_area = {np.sqrt(domain_area)}^2 m^2 vs "
            f"tile_area = {np.sqrt(tile_area)}^2 m^2)"
        )

    if domain_bounds_geometry.contains(tile_domain.spatial_bounds_geometry):
        return tile_domain
    else:
        return generate_randomly_located_tile(domain=domain, tile_size=tile_size)


def generate_tile_domain_with_peturbed_location(
    domain, tile_domain, tile_size, distance_size_scaling
):
    """
    Generate a tile of a specific `tile_size` that fits inside `domain`
    """
    domain_bounds_geometry = domain.spatial_bounds_geometry

    theta = 2 * pi * np.random.random()
    r = distance_size_scaling * tile_size
    dlx = r * np.cos(theta)
    dly = r * np.sin(theta)

    x_t = tile_domain.x_c + dlx
    y_t = tile_domain.y_c + dly

    tile_domain_perturbed = CartesianSquareTileDomain(x_c=x_t, y_c=y_t, size=tile_size)
    if isinstance(domain, LocalCartesianDomain):
        tile_domain_perturbed = tile_domain_perturbed.locate_in_latlon_domain(
            domain=tile_domain
        )

    if domain_bounds_geometry.contains(tile_domain_perturbed.spatial_bounds_geometry):
        return tile_domain_perturbed
    else:
        return generate_tile_domain_with_peturbed_location(
            domain=domain,
            tile_domain=tile_domain,
            tile_size=tile_size,
            distance_size_scaling=distance_size_scaling,
        )


def generate_triplet_location(
    domain,
    tile_size,
    neigh_dist_scaling=1.0,
):
    """
    Generate a set of (x,y)-positions (a list of three specifically)
    representing the "anchor", "neighbor" and "distant" tile locations
    """

    anchor_tile_domain = generate_randomly_located_tile(
        domain=domain, tile_size=tile_size
    )

    neighbor_tile_domain = generate_tile_domain_with_peturbed_location(
        domain=domain,
        tile_domain=anchor_tile_domain,
        tile_size=tile_size,
        distance_size_scaling=neigh_dist_scaling,
    )
    distant_tile_domain = generate_randomly_located_tile(
        domain=domain, tile_size=tile_size
    )

    return [anchor_tile_domain, neighbor_tile_domain, distant_tile_domain]
