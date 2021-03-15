"""
Pipeline specific to fetching and processing GOES-16 satellite data
"""
import datetime
from pathlib import Path

import dateutil.parser
import luigi
import numpy as np
import satdata
import xarray as xr

from ...dataset import GenericDatasource
from ....pipeline import YAMLTarget
from . import bbox, processing, satpy_rgb, tiler
from ...utils import SOURCE_DIR
