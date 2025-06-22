import scipy.ndimage
import scipy.io
import numpy as np


mech_feature_list = ['tap2', 'tap2u']
audio_feature_list = ['song', 'pulse', 'sine', 'mix', 'silence']
dist_feature_list = ['mfDist']
rot_velocity_feature_list = ['fRS', 'mRS']
orientation_feature_list = ['fTheta']
trans_velocity_feature_list = ['mFV', 'mFS', 'mLV', 'mLS', 'fFV', 'fFS', 'fLV', 'fLS']
trans_acc_feature_list = ['fFA', 'mFA']
velocity_feature_list = trans_velocity_feature_list + trans_acc_feature_list + rot_velocity_feature_list + dist_feature_list

units_dict = dict()
units_dict.update(dict.fromkeys(trans_velocity_feature_list, 'mm/s'))
units_dict.update(dict.fromkeys(trans_acc_feature_list, 'mm/s^2'))
units_dict.update(dict.fromkeys(rot_velocity_feature_list, 'deg/s'))
units_dict.update(dict.fromkeys(dist_feature_list, 'm'))
units_dict.update(dict.fromkeys(audio_feature_list, 'a.u.'))
units_dict.update(dict.fromkeys(mech_feature_list, 'a.u.'))
units_dict.update(dict.fromkeys(orientation_feature_list, 'deg'))


def split_name(e_name):
    """
    Splits event or feature name into a base name and variation specified (pos, neg, noTap, noSong)
    """
    event_name_split = e_name.split('_')
    base_event_name = event_name_split[0]
    event_var = event_name_split[1] if len(event_name_split) == 2 else ''
    print(base_event_name, event_var)
    assert base_event_name in units_dict
    assert len(event_name_split) <= 2
    return base_event_name, event_var


def connected_components1d(x, return_limits=False):
    """Return the indices of the connected components in a 1D logical array.

    Args:
        x: 1d logical (boolean) array.
        return_limits: If True, return indices of the limits of each component rather
            than every index. Defaults to False.

    Returns:
        If return_limits is False, a list of (variable size) arrays are returned, where
        each array contains the indices of each connected component.

        If return_limits is True, a single array of size (n, 2) is returned where the
        columns contain the indices of the starts and ends of each component.
    """
    L, n = scipy.ndimage.label(x.squeeze())
    ccs = scipy.ndimage.find_objects(L)
    starts = [cc[0].start for cc in ccs]
    ends = [cc[0].stop for cc in ccs]
    if return_limits:
        return np.stack([starts, ends], axis=1)
    else:
        return [np.arange(i0, i1, dtype=int) for i0, i1 in zip(starts, ends)]


def merge_intervals(intervals, max_diff):
    # assume non-overlapping intervals
    # assume intervals is sorted asc by first element

    if len(intervals) == 0:
        return []

    merged_intervals = []
    current_interval = intervals[0]
    for next_interval in intervals[1:]:
        if next_interval[0] - current_interval[1] < max_diff:
            current_interval = np.array([current_interval[0], next_interval[1]])
        else:
            merged_intervals.append(current_interval)
            current_interval = next_interval
    merged_intervals.append(np.array(current_interval))
    return np.array(merged_intervals)


def skip_intervals_too_close(intervals, gap_no_tap):
    # assume non-overlapping intervals
    # assume intervals is sorted asc by first element

    if len(intervals) == 0:
        return []

    return_intervals = []
    prev_interval = intervals[0]
    return_intervals.append(prev_interval)
    for curr_interval in intervals[1:]:
        if curr_interval[0] - prev_interval[1] < gap_no_tap:
            pass
        else:
            return_intervals.append(curr_interval)
        prev_interval = curr_interval
    return np.array(return_intervals)

def get_nanmean_angle(theta, axis):
    """
    theta: array of angles in radians

    Returns:
    -------
    mean_theta: mean angle in radians
    """
    rad_to_complex = np.exp(theta * 1j)
    mean_complex_nu = np.nanmean(rad_to_complex, axis=axis)
    mean_theta = np.angle(mean_complex_nu)
    return mean_theta


def get_nanstd_angle(theta, axis):
    """
    theta: array of angles in radians

    Returns:
    -------
    std_theta: std of angles in radians
    """
    rad_to_complex = np.exp(theta * 1j)
    std_theta = np.nanstd(rad_to_complex, axis=axis)
    return std_theta
