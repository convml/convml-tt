#!/usr/bin/env python
# coding: utf-8
"""
Module for loading images and classification from S. Rasp et al study on
hand-labelled classification of clouds

1px ~ 1km for the sugar, fish etc dataset
for the model I have trained so far I have 256 pixels covering 200km
so I need to take 200px x 200px and resample to 256x256
"""

import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from . import rolling_window

def rle2mask(rle, imgshape):
    width, height = imgshape
    mask = np.zeros(width*height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    if starts.max() > len(mask):
        raise Exception

    current_position = 0
    for i_start, l in zip(starts, lengths):
        mask[i_start:i_start+l] = 1
        current_position += l

    return np.flipud(np.rot90(mask.reshape(width, height), k=1))

def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_img_and_mask(img_fn):
    df_img = df[df.Image_Label.str.contains(img_fn)]

    masks = {}
    for i, df_row in df_img.iterrows():
        image_label = df_row['Image_Label']
        img_fn, label_type = image_label.split('_')

        img_p = path/"train_images"/img_fn
        img = Image.open(img_p)

        encoded_pixels = df_row['EncodedPixels']
        if type(encoded_pixels) == str:
            mask = rle2mask(encoded_pixels, imgshape=img.size)
        else:
            mask = np.zeros(img.size[::-1])
        masks[label_type] = mask

    masks_labels, masks_arr = zip(*masks.items())
    da_masks = xr.DataArray(
        np.dstack(masks_arr),
        dims=('y', 'x', 'label_type'),
        coords=dict(label_type=np.array(masks_labels)),
        name='mask'
    )

    da_img = xr.DataArray(img, name='img', dims=('y', 'x', 'rgb'))

    return xr.merge([da_masks, da_img])

def get_rolling_img_crops(ds_img, label_type, skip_size, window=(200, 200)):
    ds_img_cropped = ds_img.where(ds_img.sel(label_type=label_type).mask, drop=True)
    img_cropped = ds_img_cropped.img

    if img_cropped.count() == 0:
        return None

    img_windows = rolling_window.rolling_window(
        img_cropped, window=list(window)+[0,],
        asteps=(skip_size,skip_size,1), axes=(0,1)
    )

    return img_windows

def get_cropped_img(ds_img, label_type, skip_size, img_size=(256, 256)):
    img_windows = get_rolling_img_crops(ds_img=ds_img, label_type=label_type, skip_size=skip_size)

    if img_windows is None:
        return []

    nwx, nwy, _, _, _ = img_windows.shape
    for i in range(nwx):
        for j in range(nwy):
            img_arr = img_windows[i,j]
            if np.any(np.isnan(img_arr)):
                continue
            else:
                img_arr = img_arr.astype(ds_img.img.dtype)

            img_scaled = Image.fromarray(np.rot90(img_arr.transpose((2,1,0)))).resize(img_size, Image.BILINEAR)
            yield img_scaled

def get_imgs(data_path, label_size, skip_size):
    path = Path(data_path)
    df = pd.read_csv(path/"train.csv")
    df.head()

    for img_p in (path/"train_images").glob("*.jpg"):
        img_fn = img_p.name
        print(img_fn)

        ds_img = load_img_and_mask(img_fn)
        for img_scaled in get_cropped_img(ds_img, label_type, skip_size):
            yield img_scaled
