from pathlib import Path

import numpy as np
from PIL import Image

from ....data.sources.pipeline import parse_scene_id


def plot_scene_trajectories(
    ds_traj, scene_id, data_source, dt_max=np.timedelta64(2, "h"), ax=None
):
    """
    Create scene plot with trajectories overlaid showing the timespan `dt_max`
    forward and backward (in solid and transparent colors respectively)
    """
    if "time" not in ds_traj.coords:
        ds_traj["time"] = ("scene_id"), [
            parse_scene_id(scene_id)[1] for scene_id in ds_traj.scene_id.data
        ]

    ds_traj_scene = ds_traj.sel(scene_id=scene_id)
    path_image = (
        Path(data_source._meta["data_path"]) / ds_traj_scene.image_filename.item()
    )
    img_scene = Image.open(path_image)
    extent_img_scene = data_source.domain.get_grid_extent()

    if ax is None:
        ax = data_source.domain.plot_outline(alpha=0.0)
    ax.imshow(img_scene, extent=extent_img_scene)

    for traj_id in ds_traj.traj_id.values:
        ds_traj_single = ds_traj.sel(traj_id=traj_id)
        ds_point = ds_traj_single.sel(scene_id=scene_id)
        ds_traj_single_bytime = ds_traj_single.swap_dims(scene_id="time")
        t_scene = ds_traj_scene.time.data

        ds_traj_single_backward = ds_traj_single_bytime.sel(
            time=slice(t_scene - dt_max, t_scene)
        )
        ds_traj_single_forward = ds_traj_single_bytime.sel(
            time=slice(t_scene, t_scene + dt_max)
        )

        ds_parts = [ds_traj_single_backward, ds_traj_single_forward]
        kwargs_parts = [dict(), dict(alpha=0.6)]
        color = None
        for ds_part, kwargs in zip(ds_parts, kwargs_parts):
            if color is not None:
                kwargs["color"] = color
            if ds_part.scene_id.count() > 1:
                (line,) = ds_part.swap_dims(time="x").y.plot(x="x", ax=ax, **kwargs)
                color = line.get_color()

        ax.scatter(
            ds_point.x,
            ds_point.y,
            color=color,
            marker=".",
        )

    ax.margins(1.0)
    ax.set_title(scene_id)

    return ax
