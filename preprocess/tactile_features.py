"""
CAUTION!!!! Only to be used for leap rig's gold data
"""

import numpy as np
import shapely.geometry as geom
from shapely.geometry import Point

from preprocess.preproc_utils import signed_angle

FEMALE_WIDTH = 36 / 30.3    # 36 pixels female width in leap rig, 30.3 PIXEL_TO_MM for leap rig data
FEMALE_LENGTH = 90 / 30.3
MALE_WIDTH = 30 / 30.3
MALE_LENGTH = 75 / 30.3


def __rotate_points(x: np.ndarray, rad: np.ndarray):
    """Rotate points, x, around the origin by angles, rad."""
    x = np.atleast_2d(x)
    rad = np.atleast_1d(rad)
    rot = np.array([[np.cos(rad), -np.sin(rad)],
                    [np.sin(rad), np.cos(rad)]])
    if len(x) == len(rad):
        einsum = "ij...n,n...j->n...i"
    else:
        einsum = "ij...n,...j->n...i"
    return np.einsum(einsum, rot, x)


def __create_ellipse(length: float, width: float, rotation: np.ndarray, offset: np.ndarray):
    rotation = np.array(rotation)
    offset = np.array(offset)

    ellipse = np.radians(np.arange(0, 360, 1))
    ellipse = np.column_stack([np.cos(ellipse) * length / 2, np.sin(ellipse) * width / 2])
    offset = np.atleast_2d(offset)
    # print(offset.shape)
    return __rotate_points(ellipse, rotation) + offset[:, np.newaxis, :]


def __get_mech_distance(mTrx, fTrx, FLY_SKELETON):
    """

    """

    ellipse_center_female = fTrx[:, FLY_SKELETON.index('thorax')]
    ellipse_points_female = __create_ellipse(FEMALE_LENGTH, FEMALE_WIDTH, 0, ellipse_center_female)
    ellipse_female = [geom.Polygon(xy) for xy in ellipse_points_female]

    mech_distance = []
    min_male_part_idxs = []
    for i, (ep, m) in enumerate(zip(ellipse_female, mTrx)):
        male_distance = ep.distance([Point(_) for _ in m])
        min_male_part_idx = np.nanargmin(male_distance)
        min_male_distance = np.nanmin(male_distance)
        min_male_part_idxs.append(min_male_part_idx)
        mech_distance.append(min_male_distance)

    # # min_male_part_idxs = np.array(min_male_part_idxs)
    # # print(mTrx.shape, min_male_part_idxs.shape)
    # m_min_part_rotated = rotate_points(mTrx[range(len(mTrx)), min_male_part_idxs], np.radians(-fTheta)).squeeze()
    # print("m_min_part_rotated", m_min_part_rotated.shape)
    #
    # # if after rotation male y>0, then left. if <0, then right.
    # # TODO: flip the sign here to be compatible with regular convention of left (-1) and right (1).
    # mech_side = np.sign(m_min_part_rotated[:, 1])
    # print(mech_side, np.sum(mech_side), np.unique(mech_side, return_counts=True))
    return np.array(mech_distance), np.array(min_male_part_idxs)


def compute_mechanical_features_v1(mTrx, fTrx, FLY_SKELETON):

    orig_dim = fTrx.ndim
    # print("orig_dim", orig_dim)

    skeleton_size = fTrx.shape[-2]
    # print("skeleton_size", skeleton_size)
    if orig_dim == 4:
        time_dim = fTrx.shape[1]

    headIdx = FLY_SKELETON.index('head')
    abdomenIdx = FLY_SKELETON.index('abdomen')
    wingLIdx = FLY_SKELETON.index('wingL')
    wingRIdx = FLY_SKELETON.index('wingR')
    forelegLIdx = FLY_SKELETON.index('forelegL4')
    forelegRIdx = FLY_SKELETON.index('forelegR4')

    mTrx = mTrx.reshape(-1, skeleton_size, 2)
    fTrx = fTrx.reshape(-1, skeleton_size, 2)

    # print(mTrx.shape, fTrx.shape)

    mHead = mTrx[..., headIdx, :]
    mForelegL = mTrx[..., forelegLIdx, :]
    mForelegR = mTrx[..., forelegRIdx, :]

    fAbdomen = fTrx[..., abdomenIdx, :]
    fWingL = fTrx[..., wingLIdx, :]
    fWingR = fTrx[..., wingRIdx, :]

    # print(mHead.shape)

    contact_threshold = 0.2   # in mm
    bump_condition = (
            (np.sqrt(np.sum((mForelegL - fAbdomen) ** 2, axis=1)) <= contact_threshold)
            | (np.sqrt(np.sum((mForelegR - fAbdomen) ** 2, axis=1)) <= contact_threshold)
            | (np.sqrt(np.sum((mHead - fWingL) ** 2, axis=1)) <= contact_threshold)
            | (np.sqrt(np.sum((mHead - fWingR) ** 2, axis=1)) <= contact_threshold)
            | (np.sqrt(np.sum((mHead - fAbdomen) ** 2, axis=1)) <= contact_threshold)
    )

    mech_sense = np.where(bump_condition, 1, 0)
    # print("mech_sense", np.sum(mech_sense), mech_sense.shape, mech_sense)

    ftrs = dict()
    ftrs["touch"] = mech_sense

    if orig_dim == 4:
        for f in ftrs:
            ftrs[f] = ftrs[f].reshape(-1, time_dim)
    return ftrs


def compute_mechanical_features_v2(mTrx, fTrx, FLY_SKELETON):

    orig_dim = fTrx.ndim
    skeleton_size = fTrx.shape[-2]
    if orig_dim == 4:
        time_dim = fTrx.shape[1]

    mTrx = mTrx.reshape(-1, skeleton_size, 2)
    fTrx = fTrx.reshape(-1, skeleton_size, 2)

    min_distance, min_male_part_idx = __get_mech_distance(mTrx, fTrx, FLY_SKELETON)

    fHd = fTrx[:, FLY_SKELETON.index('head')]
    fThx = fTrx[:, FLY_SKELETON.index('thorax')]
    mMinPart = mTrx[range(len(mTrx)), min_male_part_idx]

    fDir = fHd - fThx   # direction from female thorax to female head
    fmpDir = fThx - mMinPart # direction from male's closest point to female head (mMinPart - fThx)
    fmpAng = signed_angle(fDir, fmpDir)

    ftrs = dict()
    ftrs["tapClosestDist"] = min_distance
    ftrs["tapClosestAng"] = fmpAng
    if orig_dim == 4:
        for f in ftrs:
            ftrs[f] = ftrs[f].reshape(-1, time_dim)
    return ftrs

