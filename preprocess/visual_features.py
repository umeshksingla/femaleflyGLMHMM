import numpy as np
from preprocess.preproc_utils import signed_angle


def compute_visual_features(fTrx, mTrx, FLY_SKELETON):
    """Extract behavioral features given head and thorax coordinates.

    Args:
        fTrx: Female track coordinates in array of shape (timesteps, b, 2).
        mTrx: Male track coordinates in array of shape (timesteps, b, 2).

    where b is len(FLY_SKELETON).
    Returns:
        A dictionary of classical features with keys:

        mfDist: Euclidean distance between the male and female thorax.
        mFV: Forward velocity - magnitude of the velocity in the direction of heading (male).
        fFV: Forward velocity - magnitude of the velocity in the direction of heading (female).
        mFA: Forward acceleration (male).
        fFA: Forward acceleration (female).
        mLV: Lateral velocity - signed magnitude of the velocity perpendicular to the forward velocity (male).
        fLV: Lateral velocity - signed magnitude of the velocity perpendicular to the forward velocity (female).
        mLS: Lateral speed - absolute magnitude of perpendicular velocity (male).
        fLS: Lateral speed - absolute magnitude of perpendicular velocity (female).
        mLA: Lateral acceleration (male).
        fLA: Lateral acceleration (female).
        mRS: Rotational speed - change in the heading (male).
        fRS: Rotational speed - change in the heading (female).
        mfAng: Angle subtended by one fly on the other fly (male to female).
        fmAng: Angle subtended by one fly on the other fly (female to male).
        mfFV: Velocity in the direction of the other fly (male towards female).
        fmFV: Velocity in the direction of the other fly (female towards male).
        mfLS: Lateral speed of fly in perpendicular direction of the other fly (male towards female).
        fmLS: Lateral speed of fly in perpendicular direction of the other fly (female towards male).

    Notes:
        Based off of Junyu Li's implementation (/tigress/MMURTHY/junyu/code/alignFeature/compute_features.py).
    """

    # Get relevant indices
    fHd = fTrx[..., FLY_SKELETON.index('head'), :]
    fThx = fTrx[..., FLY_SKELETON.index('thorax'), :]
    mHd = mTrx[..., FLY_SKELETON.index('head'), :]
    mThx = mTrx[..., FLY_SKELETON.index('thorax'), :]
    mLwing = mTrx[..., FLY_SKELETON.index('wingL'), :]
    mRwing = mTrx[..., FLY_SKELETON.index('wingR'), :]

    orig_dim = fThx.ndim
    # print("orig_dim", orig_dim)
    if orig_dim == 3:
        time_dim = fThx.shape[1]
        fThx = fThx.reshape(-1, 2)
        mThx = mThx.reshape(-1, 2)
        fHd = fHd.reshape(-1, 2)
        mHd = mHd.reshape(-1, 2)
        mLwing = mLwing.reshape(-1, 2)
        mRwing = mRwing.reshape(-1, 2)

    # Euclidean distance between the male and female thorax.
    mfDist = np.sqrt(np.sum((fThx - mThx) ** 2, axis=1))

    # Vector joining the thorax points in consecutive frames.
    mV_vec = np.diff(mThx, axis=0)
    mV_vec = np.pad(mV_vec, ((0, 1), (0, 0)), mode="edge")
    fV_vec = np.diff(fThx, axis=0)
    fV_vec = np.pad(fV_vec, ((0, 1), (0, 0)), mode="edge")

    # # Velocity - the Euclidean distance moved by the thorax in each frame.
    mV = np.sqrt(np.sum(mV_vec ** 2, axis=1))
    fV = np.sqrt(np.sum(fV_vec ** 2, axis=1))

    # Unit vector joining the head to the thorax or heading.
    mDir = mHd - mThx
    fDir = fHd - fThx
    mDir_unit = mDir / np.linalg.norm(mDir, axis=1, keepdims=True)
    fDir_unit = fDir / np.linalg.norm(fDir, axis=1, keepdims=True)

    # Angle made by the body axis with the x-axis.
    mTheta = np.rad2deg(np.arctan2(mDir[:, 1], mDir[:, 0]))
    fTheta = np.rad2deg(np.arctan2(fDir[:, 1], fDir[:, 0])) # -180/180

    # Forward velocity - magnitude of the velocity in the direction of heading.
    mFV = np.sum(mV_vec * mDir_unit, axis=1)
    mFA = np.diff(mFV, axis=0)
    mFA = np.pad(mFA, (0, 1), mode="edge")
    fFV = np.sum(fV_vec * fDir_unit, axis=1)
    fFA = np.diff(fFV, axis=0)
    fFA = np.pad(fFA, (0, 1), mode="edge")

    # Lateral velocity - magnitude of the velocity perpendicular to the forward velocity.
    mLV = np.sum(mV_vec * np.stack([-mDir_unit[:, 1], mDir_unit[:, 0]], axis=1), axis=1)
    fLV = np.sum(fV_vec * np.stack([-fDir_unit[:, 1], fDir_unit[:, 0]], axis=1), axis=1)
    #     mLV = np.abs(np.sum(mV_vec * np.stack([-mDir_unit[:, 1], mDir_unit[:, 0]], axis=1), axis=1))
    #     mLVr = np.sum(mV_vec * np.stack([-mDir_unit[:, 1], mDir_unit[:, 0]], axis=1), axis=1)
    #     mLVl = np.sum(mV_vec * np.stack([mDir_unit[:, 1], -mDir_unit[:, 0]], axis=1), axis=1)

    # # Lateral acceration.
    mLA = np.diff(mLV)
    mLA = np.pad(mLA, (0, 1), mode="edge")
    fLA = np.diff(fLV)
    fLA = np.pad(fLA, (0, 1), mode="edge")

    # Rotational speed - change in the heading of the male
    delt = 1
    mRS = signed_angle(mDir[0: (-1 - delt), :], mDir[delt:-1, :])
    mRS = np.pad(mRS, (1, 1), mode="edge")
    fRS = signed_angle(fDir[0: (-1 - delt), :], fDir[delt:-1, :])
    fRS = np.pad(fRS, (1, 1), mode="edge")

    # Angular velocity (of the thorax)
    # Vector joining one fly's thorax to the other's
    mfDir = fHd - mThx
    fmDir = mHd - fThx
    fmDir_unit = fmDir / np.linalg.norm(fmDir, axis=1, keepdims=True)
    mfDir_unit = mfDir / np.linalg.norm(mfDir, axis=1, keepdims=True)

    # # Angle subtended by one fly on the other fly
    mfAng = signed_angle(mDir, fmDir)
    fmAng = signed_angle(fDir, mfDir)

    # Velocity in the direction of the other fly.
    fmFV = np.sum(fV_vec * fmDir_unit, axis=1)
    mfFV = np.sum(mV_vec * mfDir_unit, axis=1)

    # # Male lateral speed in female direction: mfLS
    # #     mfLS = np.sqrt(mV ** 2 - mfFV ** 2)
    # #     mfLS = np.sqrt((mV - mfFV) ** 2)
    # mfDir_unit_perp = np.stack([-mfDir_unit[:, 1], mfDir_unit[:, 0]], axis=1)
    # mfLV = np.sum(mV_vec * mfDir_unit_perp, axis=1)
    #
    # # Female lateral speed in male direction: fmLS
    # #     fmLS = np.sqrt(fV ** 2 - fmFV ** 2)
    # #     fmLS = np.sqrt((fV - fmFV) ** 2)
    # fmDir_unit_perp = np.stack([-fmDir_unit[:, 1], fmDir_unit[:, 0]], axis=1)
    # fmLV = np.sum(fV_vec * fmDir_unit_perp, axis=1)

    def rotate(angle):
        """rotate counterclockwise by angle"""
        angle_rad = np.deg2rad(angle)
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

    # shift origin to fThx for the below calculations
    fHd_o = fHd - fThx
    mThx_o = mThx - fThx
    mLwing_o = mLwing - fThx
    mRwing_o = mRwing - fThx
    fThx_o = fThx - fThx

    fDir = fHd_o - fThx_o
    aristaLDir = fDir @ rotate(-70).T  # creating a line emanating from head/thorax at 70/-70 to define the arista line
    aristaRDir = fDir @ rotate(70).T    # generally rotate left is +70 but here we have opposite, as L and R are flipped in the dataset.
    # for example, fmAng is -150 when male is on the right in the video. but -ve should denote counterclockwise, i.e, on the left of the female center axis.
    # hence, flipping L and R aristae gives the right alignment angles. Check google slides for example calculations.
    mLwingDir = mLwing_o - mThx_o
    mRwingDir = mRwing_o - mThx_o

    wingLAristaLAlignAng = np.abs(signed_angle(mLwingDir, aristaLDir))  # no signed alignment required
    wingLAristaRAlignAng = np.abs(signed_angle(mLwingDir, aristaRDir))
    wingRAristaLAlignAng = np.abs(signed_angle(mRwingDir, aristaLDir))
    wingRAristaRAlignAng = np.abs(signed_angle(mRwingDir, aristaRDir))

    ftrs = dict()
    ftrs["mfDist"] = mfDist
    ftrs["mV"] = mV
    ftrs["fV"] = fV
    ftrs["mFV"] = mFV
    ftrs["fFV"] = fFV
    ftrs["mFS"] = np.abs(mFV)
    ftrs["fFS"] = np.abs(fFV)
    ftrs["mFA"] = mFA
    ftrs["fFA"] = fFA
    ftrs["mLV"] = mLV
    ftrs["fLV"] = fLV
    ftrs["mLS"] = np.abs(mLV)
    ftrs["fLS"] = np.abs(fLV)
    ftrs["mLA"] = mLA
    ftrs["fLA"] = fLA
    ftrs["mRS"] = mRS
    ftrs["fRS"] = fRS
    ftrs["mfAng"] = mfAng
    ftrs["fmAng"] = fmAng
    ftrs["mfFV"] = mfFV
    ftrs["fmFV"] = fmFV
    # ftrs["mfLV"] = mfLV
    # ftrs["fmLV"] = fmLV
    #     ftrs["mfLS"] = np.abs(mfLV)
    #     ftrs["fmLS"] = np.abs(fmLV)
    ftrs['mTheta'] = mTheta
    ftrs['fTheta'] = fTheta
    ftrs['wingLAristaLAlignAng'] = wingLAristaLAlignAng
    ftrs['wingRAristaRAlignAng'] = wingRAristaRAlignAng
    ftrs['wingLAristaRAlignAng'] = wingLAristaRAlignAng
    ftrs['wingRAristaLAlignAng'] = wingRAristaLAlignAng
    if orig_dim == 3:
        for f in ftrs:
            ftrs[f] = ftrs[f].reshape(-1, time_dim)
    return ftrs
