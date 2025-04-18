import numpy as np
import miniball
from scipy.signal import savgol_filter
import h5py


def h5read(filename, dataset):
    """Load a single dataset from HDF5 file.

    Args:
        filename: Path to HDF5 file.
        dataset: Name of the dataset.

    Returns:
        The dataset data loaded in.
    """
    with h5py.File(filename, "r") as f:
        return f[dataset][:]


def describe_hdf5(filename, attrs=True):
    """Describe all items in an HDF5 file."""

    def desc(k, v):
        if type(v) == h5py.Dataset:
            print(f"[ds]  {v.name}: {v.shape} | dtype = {v.dtype}")
            if attrs and len(v.attrs) > 0:
                print(f"      attrs = {dict(v.attrs.items())}")
        elif type(v) == h5py.Group:
            print(f"[grp] {v.name}:")
            if attrs and len(v.attrs) > 0:
                print(f"      attrs = {dict(v.attrs.items())}")

    with h5py.File(filename, "r") as f:
        f.visititems(desc)


def lims_to_mask(lims, size=None):
    """Convert limits to a mask.

    Args:
        lims: Indices of limits as an array of shape (n, 2).
        size: Number of elements in the vector. If not provided, uses the largest limit.

    Returns:
        A logical vector of shape (size,) where elements whose indices are contained in
        the input lims are True.
    """
    if not isinstance(lims, np.ndarray):
        lims = np.array(lims)
    lims = lims.astype(int)
    if size is None:
        size = lims.max()
    mask = np.full((size,), False)
    for i0, i1 in lims:
        mask[i0:i1] = True
    return mask


def signed_angle(a, b):
    """Finds the signed angle between two 2D vectors a and b.

    Args:
        a: Array of shape (n, 2).
        b: Array of shape (n, 2).

    Returns:
        The signed angles in degrees in vector of shape (n, 2).

        This angle is positive if a is rotated clockwise to align to b and negative if
        this rotation is counter-clockwise.
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    theta = np.arccos(np.around(np.sum(a * b, axis=1), decimals=4))
    cross = np.cross(a, b, axis=1)
    sign = np.zeros(cross.shape)
    sign[cross >= 0] = -1
    sign[cross < 0] = 1
    return np.rad2deg(theta) * sign


def smooth(x, smooth_window):
    return savgol_filter(x, window_length=smooth_window, polyorder=1, axis=0)


def circle_estimator_helper(trxM, trxF):
    """
    """

    fr, malexminbodypart = np.unravel_index(np.nanargmin(trxM[..., 0]), trxM[..., 0].shape)
    malexmin = trxM[fr]
    fr, malexmaxbodypart = np.unravel_index(np.nanargmax(trxM[..., 0]), trxM[..., 0].shape)
    malexmax = trxM[fr]
    fr, maleyminbodypart = np.unravel_index(np.nanargmin(trxM[..., 1]), trxM[..., 1].shape)
    maleymin = trxM[fr]
    fr, maleymaxbodypart = np.unravel_index(np.nanargmax(trxM[..., 1]), trxM[..., 1].shape)
    maleymax = trxM[fr]

    fr, femalexminbodypart = np.unravel_index(np.nanargmin(trxF[..., 0]), trxF[..., 0].shape)
    femalexmin = trxF[fr]
    fr, femalexmaxbodypart = np.unravel_index(np.nanargmax(trxF[..., 0]), trxF[..., 0].shape)
    femalexmax = trxF[fr]
    fr, femaleyminbodypart = np.unravel_index(np.nanargmin(trxF[..., 1]), trxF[..., 1].shape)
    femaleymin = trxF[fr]
    fr, femaleymaxbodypart = np.unravel_index(np.nanargmax(trxF[..., 1]), trxF[..., 1].shape)
    femaleymax = trxF[fr]

    points = [malexmin[malexminbodypart], femalexmin[femalexminbodypart],
              malexmax[malexmaxbodypart], femalexmax[femalexmaxbodypart],
              maleymin[maleyminbodypart], femaleymin[femaleyminbodypart],
              maleymax[maleymaxbodypart], femaleymax[femaleymaxbodypart]
              ]

    points = np.array(points)
    c, r2 = miniball.get_bounding_ball(points, epsilon=1e-12, rng=np.random.default_rng(42))
    r = np.sqrt(r2)
    return (c[1], c[0]), r
