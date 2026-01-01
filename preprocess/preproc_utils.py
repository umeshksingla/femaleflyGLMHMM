import numpy as np
import miniball
import h5py
import scipy.ndimage
from scipy.signal import savgol_coeffs, lfilter
from scipy.stats import zscore
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


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


def safe_zscore(x):
    std_dev = np.std(x)
    if np.isclose(std_dev, 0, atol=1e-2):
        return np.zeros_like(x)
    return zscore(x)


def halfgaussian_kernel(sigma, radius):
    """
    Computes a Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    # The half-gaussian is just the right side of the bell curve
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x


def halfgaussian_filter(input, sigma, axis=0, output=None,
                      mode="nearest", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to 'truncate' standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)


def smooth_savgol_noncausal(x, window_length):
    """
    Smoothes using Savitzky-Golay filter. Uses future data points to smooth.
    """
    return savgol_filter(x, window_length=window_length, polyorder=1, axis=0)


def smooth_savgol_causal(data, window_length):
    """
    Smoothes raw tracks using a causal Savitzky-Golay filter. No future data is used.
    """

    # Get coefficients for the CURRENT point (pos = the edge of the window)
    coeffs = savgol_coeffs(window_length, polyorder=1, deriv=0, pos=window_length-1)

    # Apply lfilter. Equivalent to a weighted moving average of the last 'window_length' points
    smoothed_signal = lfilter(coeffs[::-1], [1.0], data, axis=0)

    # Set the first (window_length-1) points to the raw data or NaN as the filter hasn't filled its buffer yet.
    smoothed_signal[:window_length - 1] = data[:window_length - 1]
    return smoothed_signal


def fill_missing_tracks_SR(Y, kind="linear"):
    """
    Methods to interpolate missing tracking points using scipy.interpolate.interp1d
    A mixture of linear and cubic spline interpolators are used.

    Code by Shruthi Ravindranathan.
    """
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))
    # print(f"\tY_initial.shape = {initial_shape}, Y.shape={Y.shape}")
    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        non_missing_mask = ~np.isnan(y)
        num_non_missing = np.sum(non_missing_mask)

        # If we don't have enough points, don't interpolate, if we only have 2-3,
        # use linear interpolation, otherwise, use the interpolation requested.
        if num_non_missing <= 1:
            continue
        elif num_non_missing <= 4:
            kind_i = "linear"
        else:
            kind_i = kind

        # Build interpolant.
        x = np.flatnonzero(non_missing_mask)
        f = interp1d(x, y[x], kind=kind_i, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        Y[xq, i] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(Y[:, i])
        Y[:, i][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Y[:, i][~mask])

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)
    return Y


def circle_estimator_helper(trxM, trxF):
    """
    Estimate approximate chamber radius and center coordinates by building the smallest bounding ball around extreme
    fly tracking coordinates. Used to approximate the distance of flies from the chamber walls/edges at any time.
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
