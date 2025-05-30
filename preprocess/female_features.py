import matplotlib.pyplot as plt
import numpy as np
from preprocess.preproc_utils import signed_angle


def normalize_to_egocentric(x, rel_to=None, scale_factor=1, ctr_ind=1, fwd_ind=0, return_angles=False):
    """Normalize pose estimates to egocentric coordinates.

    Args:
        x: Pose of shape (joints, 2) or (time, joints, 2)
        rel_to: Pose to align x with of shape (joints, 2) or (time, joints, 2). Defaults
            to x if not specified.
        scale_factor: Spatial scaling to apply to coordinates after centering.
        ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.
        fill: If True, interpolate missing ctr and fwd coordinates. If False, timesteps
            with missing coordinates will be all NaN. Defaults to True.
        return_angles: If True, return angles with the aligned coordinates.

    Returns:
        Egocentrically aligned poses of the same shape as the input.

        If return_angles is True, also returns a vector of angles.
    """

    if rel_to is None:
        rel_to = x

    is_singleton = (x.ndim == 2) and (rel_to.ndim == 2)

    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    if rel_to.ndim == 2:
        rel_to = np.expand_dims(rel_to, axis=0)

    # Find egocentric forward coordinates.
    ctr = rel_to[..., ctr_ind, :]  # (t, 2)
    fwd = rel_to[..., fwd_ind, :]  # (t, 2)
    ego_fwd = fwd - ctr

    # Compute angle.
    ang = np.arctan2(ego_fwd[..., 1], ego_fwd[..., 0])  # arctan2(y, x) -> radians in [-pi, pi]
    ca = np.cos(ang)  # (t,)
    sa = np.sin(ang)  # (t,)

    # Build rotation matrix.
    rot = np.zeros([len(ca), 3, 3], dtype=ca.dtype)
    rot[..., 0, 0] = ca
    rot[..., 0, 1] = -sa
    rot[..., 1, 0] = sa
    rot[..., 1, 1] = ca
    rot[..., 2, 2] = 1

    # Center and scale.
    x = x - np.expand_dims(ctr, axis=1)
    x /= scale_factor

    # Pad, rotate and crop.
    x = np.pad(x, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1) @ rot
    x = x[..., :2]

    if is_singleton:
        x = x[0]

    if return_angles:
        return x, ang
    else:
        return x


def compute_wing_angles(x, left_ind=3, right_ind=4):
    """Returns the wing angles in degrees from normalized pose.

    Args:
        x: Egocentric pose of shape (..., joints, 2). Use normalize_to_egocentric on the
            raw pose coordinates before passing to this function.
        left_ind: Index of the left wing. Defaults to 3.
        right_ind: Index of the right wing. Defaults to 4.

    Returns:
        Tuple of (thetaL, thetaR) containing the left and right wing angles.

        Both are in the range [-180, 180], where 0 is when the wings are exactly aligned
        to the midline (thorax to head axis).

        Positive angles denote extension away from the midline in the direction of the
        wing. For example, a right wing extension may have thetaR > 0.
    """
    xL, yL = x[..., left_ind, 0], x[..., left_ind, 1]
    xR, yR = x[..., right_ind, 0], x[..., right_ind, 1]

    # xL, yL, xR, yR = np.abs(xL), np.abs(yL), np.abs(xR), np.abs(yR)
    thetaL = np.rad2deg(np.arctan2(yL, xL)) + 180
    thetaL[np.greater(thetaL, 180, where=np.isfinite(thetaL))] -= 360
    thetaR = np.rad2deg(np.arctan2(yR, xR)) + 180
    thetaR[np.greater(thetaR, 180, where=np.isfinite(thetaR))] -= 360
    # thetaR = -thetaR
    return thetaL, thetaR


def compute_wing_flick_features(fTrx, FLY_SKELETON):

    fThx = fTrx[..., FLY_SKELETON.index('thorax'), :]
    orig_dim = fThx.ndim
    if orig_dim == 3:
        time_dim = fThx.shape[1]

    egoF = normalize_to_egocentric(fTrx, ctr_ind=FLY_SKELETON.index('thorax'), fwd_ind=FLY_SKELETON.index('head'))
    wingFL, wingFR = compute_wing_angles(egoF, left_ind=FLY_SKELETON.index('wingL'), right_ind=FLY_SKELETON.index('wingR'))
    wingMaxAngle = np.max(np.vstack([np.abs(wingFL), np.abs(wingFR)]), axis=0)

    ftrs = dict()
    ftrs['wingFL'] = wingFL
    ftrs['wingFR'] = wingFR
    ftrs['wingMaxAngle'] = wingMaxAngle
    ftrs['wingFlick'] = np.where(
        ((np.sign(wingFL) != np.sign(wingFR)) &
        (wingMaxAngle >= 10) & (wingMaxAngle <= 30)),
        np.ones_like(wingFL), np.zeros_like(wingFL))

    if orig_dim == 3:
        for f in ftrs:
            ftrs[f] = ftrs[f].reshape(-1, time_dim)
    return ftrs


def compute_fending_features(fTrx, FLY_SKELETON):
    fThx = fTrx[..., FLY_SKELETON.index('thorax'), :]
    fmidlegL = fTrx[..., FLY_SKELETON.index('midlegL4'), :]
    fmidlegR = fTrx[..., FLY_SKELETON.index('midlegR4'), :]
    orig_dim = fThx.ndim
    if orig_dim == 3:
        time_dim = fThx.shape[1]

    ftrs = dict()
    legExtL = np.sqrt(np.sum((fThx - fmidlegL) ** 2, axis=1))
    legExtR = np.sqrt(np.sum((fThx - fmidlegR) ** 2, axis=1))
    ftrs['legExtL'] = legExtL
    ftrs['legExtR'] = legExtR

    if orig_dim == 3:
        for f in ftrs:
            ftrs[f] = ftrs[f].reshape(-1, time_dim)
    return ftrs
