import satdata
import types
import datetime


DATETIME_ATTRS = [
    attr_name
    for (attr_name, attr) in vars(datetime.datetime).items()
    if isinstance(attr, types.GetSetDescriptorType)
] + [
    attr_name
    for (attr_name, attr) in vars(datetime.date).items()
    if isinstance(attr, types.GetSetDescriptorType)
]


def within_attr_values(time, **attr_values):
    """
    Check that a give datetime attribute's value is in a list of provided values

    >> with_attr_values(time, month=[11,12])
    """
    for attr, values in attr_values.items():
        if not getattr(time, attr) in values:
            return False
    return True


def within_dt_from_zenith(time, dt_zenith_max, lon_zenith):
    t_zenith = satdata.calc_nearest_zenith_time_at_loc(lon_zenith, t_ref=time)
    dt_to_zenith = time - t_zenith
    return abs(dt_to_zenith) < dt_zenith_max
