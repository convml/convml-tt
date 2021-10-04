import satdata


def within_months(time, months):
    return time.month in months


def within_dt_from_zenith(time, dt_zenith_max, lon_zenith):
    t_zenith = satdata.calc_nearest_zenith_time_at_loc(lon_zenith, t_ref=time)
    dt_to_zenith = time - t_zenith
    return abs(dt_to_zenith) < dt_zenith_max
