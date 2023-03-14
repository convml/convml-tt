#!/usr/bin/env python
# coding: utf-8
"""
Routines for extract trajectoriew from Cartesian projected satellite images
using optical-flow methods. Much of this code is based on
[pysteps](https://github.com/pySTEPS/pysteps) wrappers to the OpenCV python
interface
"""
import cv2
import numpy as np
import xarray as xr
from numpy.ma.core import MaskedArray
from PIL import Image
from skimage.color import rgb2gray, rgba2rgb

MIN_CORNERS = 100


def shitomasi_detection(
    input_image,
    max_corners=1000,
    quality_level=0.01,
    min_distance=10,
    block_size=5,
    buffer_mask=5,
    use_harris=False,
    k=0.04,
):
    input_image = input_image.copy()

    if input_image.ndim != 2:
        raise ValueError("input_image must be a two-dimensional array")

    # Check if a MaskedArray is used. If not, mask the ndarray
    if not isinstance(input_image, MaskedArray):
        input_image = np.ma.masked_invalid(input_image)

    np.ma.set_fill_value(input_image, input_image.min())

    # buffer the quality mask to ensure that no vectors are computed nearby
    # the edges of the radar mask
    mask = np.ma.getmaskarray(input_image).astype("uint8")
    if buffer_mask > 0:
        mask = cv2.dilate(
            mask, np.ones((int(buffer_mask), int(buffer_mask)), np.uint8), 1
        )
        input_image[mask] = np.ma.masked

    # scale image between 0 and 255
    im_min = input_image.min()
    im_max = input_image.max()
    if im_max - im_min > 1e-8:
        input_image = (input_image.filled() - im_min) / (im_max - im_min) * 255
    else:
        input_image = input_image.filled() - im_min

    # convert to 8-bit
    input_image = np.ndarray.astype(input_image, "uint8")
    mask = (-1 * mask + 1).astype("uint8")

    params = dict(
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=use_harris,
        k=k,
    )

    points = cv2.goodFeaturesToTrack(input_image, mask=mask, **params)
    if points is None:
        points = np.empty(shape=(0, 2))
    else:
        points = points[:, 0, :]

    return points


class NoPointsFoundException(Exception):
    pass


def track_features(
    prvs_image,
    next_image,
    points,
    winsize=(50, 50),
    nr_levels=3,
    criteria=(3, 10, 0),
    flags=0,
    min_eig_thr=1e-4,
):
    prvs_img = prvs_image.copy()
    next_img = next_image.copy()
    p0 = np.copy(points)

    # Check if a MaskedArray is used. If not, mask the ndarray
    if not isinstance(prvs_img, MaskedArray):
        prvs_img = np.ma.masked_invalid(prvs_img)
    np.ma.set_fill_value(prvs_img, prvs_img.min())

    if not isinstance(next_img, MaskedArray):
        next_img = np.ma.masked_invalid(next_img)
    np.ma.set_fill_value(next_img, next_img.min())

    # scale between 0 and 255
    im_min = prvs_img.min()
    im_max = prvs_img.max()
    if im_max - im_min > 1e-8:
        prvs_img = (prvs_img.filled() - im_min) / (im_max - im_min) * 255
    else:
        prvs_img = prvs_img.filled() - im_min

    im_min = next_img.min()
    im_max = next_img.max()
    if im_max - im_min > 1e-8:
        next_img = (next_img.filled() - im_min) / (im_max - im_min) * 255
    else:
        next_img = next_img.filled() - im_min

    # convert to 8-bit
    prvs_img = np.ndarray.astype(prvs_img, "uint8")
    next_img = np.ndarray.astype(next_img, "uint8")

    # Lucas-Kanade
    # TODO: use the error returned by the OpenCV routine
    params = dict(
        winSize=winsize,
        maxLevel=nr_levels,
        criteria=criteria,
        flags=flags,
        minEigThreshold=min_eig_thr,
    )
    p1, status, __ = cv2.calcOpticalFlowPyrLK(prvs_img, next_img, p0, None, **params)

    # set to nan where features weren't found
    p1[status.squeeze() == 0] = np.nan

    return p1


def extract_trajectories(
    image_filenames, point_method_kwargs={}, flow_method_kwargs={}
):
    """
    Times where no trajectory point are found will have index -1
    """
    traj_points = []
    traj_files = []
    img_prev_arr = None

    for fn in image_filenames:
        img_src = Image.open(fn)
        img_gray_arr = np.array(rgb2gray(rgba2rgb(np.array(img_src))))
        if img_prev_arr is None:
            points = shitomasi_detection(img_gray_arr, **point_method_kwargs)
            if len(points) < MIN_CORNERS:
                continue
            else:
                traj_points.append(points)
                traj_files.append(fn)
        else:
            points = traj_points[-1]
            xy_end = track_features(
                img_prev_arr, img_gray_arr, points=points, **flow_method_kwargs
            )
            traj_points.append(xy_end)
            traj_files.append(fn)

        img_prev_arr = img_gray_arr
    traj_points = np.array(traj_points)

    # NB: images use (x,y) indexing order rather than (i,j)
    ny_img, nx_img = img_gray_arr.shape

    # make all invalid points have index -1 and round to nearest integer
    np.nan_to_num(traj_points, copy=False, nan=-1.0)
    traj_points = traj_points.round().astype(int)

    # y-indexing is from top-left for images
    traj_points[..., 1] = ny_img - 1 - traj_points[..., 1]

    # now we need to clean up the points. Some will be outside the valid range
    # [0, Nx] and [0, Ny]
    m_invalid = np.logical_or(
        # out of x index bounds
        np.logical_or(traj_points[..., 0] < 0, traj_points[..., 0] >= nx_img),
        # out of y index bounds
        np.logical_or(traj_points[..., 1] < 0, traj_points[..., 1] >= ny_img),
    )
    # make these all -1 (indicating invalid point)
    traj_points[..., 0] = np.where(m_invalid, -1, traj_points[..., 0])
    traj_points[..., 1] = np.where(m_invalid, -1, traj_points[..., 1])

    N_img, N_trajs, _ = traj_points.shape

    assert len(traj_files) == N_img

    ds = xr.Dataset(coords=dict(image_filename=traj_files, traj_id=np.arange(N_trajs)))
    ds["i"] = ("image_filename", "traj_id"), traj_points[..., 0]
    ds["i"].attrs["long_name"] = "x-index"
    ds["i"].attrs["units"] = "1"
    ds["j"] = ("image_filename", "traj_id"), traj_points[..., 1]
    ds["j"].attrs["long_name"] = "j-index"
    ds["j"].attrs["units"] = "1"
    return ds
