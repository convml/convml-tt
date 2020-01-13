import datetime

import satdata

from ...dataset import TripletDataset


class SatelliteTripletDataset(TripletDataset):
    def __init__(self, domain_bbox, tile_size, tile_N, channels=[1,2,3],
                 **kwargs):
        """
        tile_size: size of tile [m]
        tile_N: dimension of tile [1]
        """
        super().__init__(**kwargs)
        self.domain_bbox = domain_bbox
        self.tile_size = tile_size
        self.tile_N = tile_N
        self.channels = channels


class FixedTimeRangeSatelliteTripletDataset(SatelliteTripletDataset):
    def __init__(self, t_start, N_days, N_hours_from_zenith,
                 **kwargs):
        super().__init__(**kwargs)
        self.t_start = t_start
        self.N_days = N_days
        self.N_hours_from_zenith = N_hours_from_zenith

        lon_zenith = kwargs['domain_bbox'][1][0]

        self._dt_max = datetime.timedelta(hours=N_hours_from_zenith)
        t_zenith = satdata.calc_nearest_zenith_time_at_loc(lon_zenith, t_ref=t_start) 
        self._times = [t_zenith + datetime.timedelta(days=n) for n in range(1, N_days)]
