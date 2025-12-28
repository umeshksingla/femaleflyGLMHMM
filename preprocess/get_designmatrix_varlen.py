"""


Written by Claude AI.
"""


import gc
import joblib
import numpy as np
from pathlib import Path
from scipy.linalg import block_diag
import time

from scipy.stats import zscore
import scipy
from collections import OrderedDict
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import scipy.ndimage
from scipy.linalg import block_diag

from glm_utils.preprocessing import BasisProjection
from glm_utils.bases import identity, raised_cosine, multifeature_basis
import matplotlib.pyplot as plt

from preprocess.leaprig import WT_DATA, AC_BOTH
from preprocess.new16mic import FREDCLEANED_DATA


def transform_single_input(inp, basis, input_raw_each_dim):
    transformed = [
        BasisProjection(basis).transform(x)
        for x in inp.reshape(-1, input_raw_each_dim)
    ]
    return np.array(transformed).reshape(-1)


def transform_single_session(s_i, s, basis, input_raw_each_dim):
    print(s_i)
    return np.array([
        transform_single_input(inp, basis, input_raw_each_dim)
        for inp in s
    ])


# def smooth_moving_average(x, smooth_window):
#     return np.convolve(x, np.ones(smooth_window), 'valid') / smooth_window
#
#
# def smooth_savgol(x, smooth_window):
#     return savgol_filter(x, window_length=smooth_window, polyorder=1, axis=0)


def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x


def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)


def smooth_gaussian(x, sigma):
    return halfgaussian_filter1d(x, sigma=sigma, mode='nearest')


def safe_zscore(x):
    std_dev = np.std(x)
    if np.isclose(std_dev, 0, atol=1e-2):
        return np.zeros_like(x)
    return zscore(x)


def create_x_and_y_windows(length, x_size=1, y_size=1, x_overlap=1, y_gap_size=0):
    """

    :param length:
    :param x_size:
    :param y_size:
    :param x_overlap:
    :param y_gap_size:
    :return:
    """
    assert x_overlap >= 0
    assert x_size > 0
    assert y_size > 0

    x_idx_windows = []
    for _ in np.arange(0, length, x_overlap):
        x_idx_windows.append(np.arange(_, _+x_size))
    x_idx_windows = np.array(x_idx_windows)
    x_idx_windows = x_idx_windows[x_idx_windows[:, -1] < length-y_size-y_gap_size]
    # print("x_idx_windows", x_idx_windows, x_idx_windows.shape)

    y_idx_windows = []
    for _ in (x_idx_windows[:, -1] + y_gap_size + 1):   # y_windows are the y_size windows after the last x in each window + any gap if you want
        y_idx_windows.append(np.arange(_, _+y_size))
    y_idx_windows = np.array(y_idx_windows)
    # print("y_idx_windows", y_idx_windows, y_idx_windows.shape)
    return x_idx_windows, y_idx_windows


def get_input_feat(sessions_features, s, f_name):
    sf = sessions_features[s]
    if f_name in ['mFV', 'mLS', 'mFA', 'mLA', 'mLV', 'mfDist', 'fDistWall', 'fFV', 'fLS', 'fLV']:
        ts = sf[f_name]
        ts = smooth_gaussian(ts, sigma=3)
        feat = zscore(ts)
    elif f_name in ['song', 'sine_i', 'pulse_i', 'song_i', 'tap2']:

        # print(f_name, np.mean(sf[f_name]), np.std(sf[f_name]))

        # fig, ax = plt.subplots(2, 1, figsize=(20, 6), sharex=True)
        # ax[0].plot(sf[f_name], label=f'{f_name}')
        # ax[1].plot(safe_zscore(sf[f_name]), label=f'z-{f_name}')
        # ax[0].legend()
        # ax[1].legend()
        # plt.suptitle(s)
        # plt.tight_layout()
        # plt.show()

        feat = safe_zscore(sf[f_name]).astype(float)
    elif f_name in ['fmAng_cos']:
        ts = np.radians(np.abs(sf['fmAng']))
        ts = smooth_gaussian(ts, sigma=3)
        feat = zscore(np.cos(ts))    # cos: front to back
    elif f_name in ['fmAng_sin']:
        ts = np.radians(sf['fmAng'])
        ts = smooth_gaussian(ts, sigma=3)
        feat = zscore(np.sin(ts))    # sin: left or right of the fly
    elif f_name in ['wingAlign_song_i_directedlr2']:
        ts = np.min([sf['wingLAristaLAlignAng'],
                       sf['wingRAristaRAlignAng'],
                       sf['wingLAristaRAlignAng'],
                       sf['wingRAristaLAlignAng']], axis=0)     # positive
        ts = np.cos(np.radians(np.abs(ts))) * sf['song_i'] * np.sign(np.sin(np.radians(sf['fmAng'])))   # we do not expect the variance to be very high among these vars. so no need to zscore them separately
        # cos makes 0 deg to 1 (best alignment) and 180 degrees to -1 (worst alignment).
        # Multiplied by binary song, when song is 0, this variable is 0 (so equivalent to the worst alignment).
        # Multiplied by -1/1 direction, cos gets divided into negative and positive.
        ts = smooth_gaussian(ts, sigma=3)
        feat = safe_zscore(ts)     # it is okay to zscore these alignment angles as if they are being treated linearly
    elif f_name in ['song_directedlr', 'sine_i_directedlr', 'pulse_i_directedlr', 'song_i_directedlr', 'tap2_directedlr']:  # directedlr using male position from fmAng
        f_name_ = f_name.split('_directed')[0]
        male_pos = sf['fmAng']
        feat = sf[f_name_] * np.sign(np.sin(np.radians(male_pos)))
        feat = safe_zscore(feat)
    elif f_name in ['song_directedlr2', 'sine_i_directedlr2', 'pulse_i_directedlr2', 'song_i_directedlr2']:  # directedlr using singing wing position
        f_name_ = f_name.split('_directed')[0]
        male_pos = np.where(np.abs(sf['wingML']) > np.abs(sf['wingMR']), sf['fmWLAng'], sf['fmWRAng'])  # use the position of the most extended wing
        feat = sf[f_name_] * np.sign(np.sin(np.radians(male_pos)))
        feat = safe_zscore(feat)
    elif f_name in ['tap2_directedlr2']:    # directedlr using male position from fmAng
        f_name_ = f_name.split('_directed')[0]
        male_pos = sf['fmAng']
        feat = sf[f_name_] * np.sign(np.sin(np.radians(male_pos)))
        feat = safe_zscore(feat)
    elif f_name in ['mFV_directedlr2', 'mLS_directedlr2', 'mfDist_directedlr2', 'fFV_directedlr2', 'fLS_directedlr2']:
        f_name_ = f_name.split('_directed')[0]
        feat = zscore(sf[f_name_]) * zscore(np.sin(np.radians(sf['fmAng'])))
        feat = zscore(feat)
    else:
        raise Exception(f'unsupported {f_name} input feature.')
    # print(f_name, "done.")
    nan_counts = np.sum(np.isnan(feat))
    if nan_counts:
        print("!!!!! feat NAN", f_name, f"{nan_counts}/{len(feat)}", "NANS !!!!!")
    return feat


def get_aux_feat(sessions_features, s, f_name, aux_windows):
    sf = sessions_features[s]
    if f_name in ['mFV', 'mFS', 'mLS', 'mLV', 'mFA', 'mfDist', 'fFV', 'fFS', 'fLS', 'fLV']:
        ts = sf[f_name]
        ts = smooth_gaussian(ts, sigma=3)
        mn = ts.mean()
        std = ts.std()
        feat = np.mean(zscore(ts)[aux_windows], axis=1)
    elif f_name in ['pulse_i', 'sine_i', 'tap2']:
        ts = sf[f_name]
        mn = ts.mean()
        std = ts.std()
        feat = np.mean(safe_zscore(ts)[aux_windows], axis=1)
    elif f_name in ['fmAng_cos']:
        ts_abs = np.abs(sf['fmAng'])
        ts_rad = np.radians(ts_abs)
        ts_smoothed = smooth_gaussian(ts_rad, sigma=3)
        ts = np.cos(ts_smoothed)  # cos: front (180deg) to back (0deg)
        mn = ts.mean()
        std = ts.std()
        feat = np.mean(zscore(ts)[aux_windows], axis=1)
    elif f_name in ['fmAng_sin']:
        ts_rad = np.radians(sf['fmAng'])
        ts_smoothed = smooth_gaussian(ts_rad, sigma=3)
        ts = np.sin(ts_smoothed)  # sin: left <-> right
        mn = ts.mean()
        std = ts.std()
        feat = np.mean(zscore(ts)[aux_windows], axis=1)
    elif f_name in ['wingAlign']:
        ts = np.min([sf['wingLAristaLAlignAng'],
                       sf['wingRAristaRAlignAng'],
                       sf['wingLAristaRAlignAng'],
                       sf['wingRAristaLAlignAng']], axis=0)     # positive
        ts = smooth_gaussian(ts, sigma=3)
        mn = ts.mean()
        std = ts.std()
        feat = np.mean(zscore(ts)[aux_windows], axis=1)
    else:
        raise Exception(f'unsupported {f_name} aux feature.')
    return feat, mn, std


# def wavelet_denoise(signal):
#     coeffs = pywt.wavedec(signal, 'db4', level=4)
#     print(len(coeffs), [len(_) for _ in coeffs])
#     threshold = 0.25 * np.max(coeffs[-1])  # Set small wavelet coefficients to zero
#     coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
#     denoised_signal = pywt.waverec(coeffs, 'db4')
#     return denoised_signal


def get_output_feat(sessions_features, s, f_name, output_windows):
    sf = sessions_features[s]
    if f_name in ['fFV', 'fFS', 'fLS', 'fLV', 'fFA', 'mFV', 'mFS', 'mLS', 'mLV']:
        ts = sf[f_name]
        ts = smooth_gaussian(ts, sigma=3)
        mn = ts.mean()
        std = ts.std()
        f = np.mean(zscore(ts)[output_windows], axis=1)
    elif f_name in ['fAV']:
        ts = sf['fTheta']
        ts = smooth_gaussian(ts, sigma=3)
        fTheta = ts[output_windows]
        dfTheta = fTheta[:, -1] - fTheta[:, 0]
        dfTheta = np.where(np.abs(dfTheta) > 90, 0, dfTheta)
        mn = dfTheta.mean()
        std = dfTheta.std()
        dfTheta = zscore(dfTheta)
        f = dfTheta
    elif f_name in ['fAS']:
        ts = sf['fTheta']
        ts = smooth_gaussian(ts, sigma=3)
        fTheta = ts[output_windows]
        dfTheta_abs = np.abs(fTheta[:, -1] - fTheta[:, 0])
        dfTheta_abs = np.where(dfTheta_abs > 90, 0, dfTheta_abs)
        mn = dfTheta_abs.mean()
        std = dfTheta_abs.std()
        dfTheta_abs = zscore(dfTheta_abs)
        f = dfTheta_abs
    elif f_name in ['mAV']:
        ts = sf['mTheta']
        ts = smooth_gaussian(ts, sigma=3)
        mTheta = ts[output_windows]
        dmTheta = mTheta[:, -1] - mTheta[:, 0]
        dmTheta = np.where(np.abs(dmTheta) > 90, 0, dmTheta)
        mn = dmTheta.mean()
        std = dmTheta.std()
        dmTheta = zscore(dmTheta)
        f = dmTheta
    elif f_name in ['dfmAng']:
        ts = sf['fmAng']
        ts = smooth_gaussian(ts, sigma=3)
        fmAng = np.abs(ts)     # abs to make left and right male positions symmetric
        dfmAng = np.diff(fmAng, prepend=fmAng[0])[output_windows]
        dfmAng = np.mean(dfmAng, axis=1)             # signed changes in orientation
        mn = dfmAng.mean()
        std = dfmAng.std()
        f = zscore(dfmAng)
    elif f_name in ['dfmAng_abs']:
        ts = sf['fmAng']
        ts = smooth_gaussian(ts, sigma=3)
        fmAng = np.abs(ts)     # abs to make left and right symmetric
        dfmAng = np.diff(fmAng, prepend=fmAng[0])[output_windows]
        dfmAng_abs = np.abs(np.mean(dfmAng, axis=1))     # unsigned changes in orientation
        mn = dfmAng_abs.mean()
        std = dfmAng_abs.std()
        f = zscore(dfmAng_abs)
    # elif f_name in ['wingFlickTheta']:
    #     ts = sf.get('wingFlickAngle', sf.get('wingMaxAngle'))
    #     ts = smooth_gaussian(ts, sigma=3)
    #     wingFlickTheta = ts * sf['wingFlick']
    #     wingFlickTheta = np.mean(wingFlickTheta[output_windows], axis=1)    # mean can be taken for these angles as they are bounded between 10 and 30 degrees
    #     mn = 0  # no zscoring for wing flick angles, as most of them are zeros
    #     std = 1
    #     f = wingFlickTheta
    elif f_name in ['wingFlickBin']:
        wing_sep = sf['wingFL'] - sf['wingFR']      # right wing angle is generally -ve
        ts = (np.abs(wing_sep) > 20).astype(int)    # 20 degrees difference between left and right wing extensions
        wingFlick = (np.sum(ts[output_windows], axis=1) >= 1).astype(float)
        mn = 0
        std = 1
        f = wingFlick
    else:
        raise Exception(f'unsupported {f_name} output feature.')
    # print(f_name, f.shape)
    # np.random.shuffle(f)
    return f, mn, std


def some_plots(b_multi, basis_ortho, inputs_raw, inputs, input_raw_each_dim, input_each_dim, basis, basis_transformed, x_labels):

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(b_multi)
    ax[0].set_title('Basis')
    ax[1].plot(basis_ortho)
    ax[1].set_title('Ortho-normalized Basis')
    plt.tight_layout()
    plt.show()
    plt.close()

    for _ in range(5):
        random_session = np.random.choice(len(inputs_raw))
        idxs = np.random.choice(inputs_raw[random_session].shape[0], 10)
        fig, ax = plt.subplots(3, 1, figsize=(17, 10))

        ax[0].plot(inputs_raw[random_session][idxs].T, '-')
        ax[0].set_title(f'Raw input series ({basis_transformed})')

        ax[1].plot(inputs[random_session][idxs].T, '.-')
        ax[1].set_title(f'Basis transformed series ({basis_transformed})')

        ax[2].plot(BasisProjection(basis).inverse_transform(inputs[random_session][idxs]).T, '.-')
        ax[2].set_title(f'Basis inverse-transformed series ({basis_transformed})')

        # plot vertical lines
        c = 0
        for _ in x_labels:
            ax[0].axvline(c * input_raw_each_dim, ls=':', c='k')
            ax[1].axvline(c * input_each_dim, ls=':', c='k')
            ax[2].axvline(c * input_raw_each_dim, ls=':', c='k')
            ax[0].text(c * input_raw_each_dim + 1, 0.1, _, color='r', rotation=90)
            c += 1

        plt.tight_layout()
        plt.show()
        plt.close()
    return


def some_plotsb3(b3, b_multi, inputs_raw, inputs, inputs_multi, input_raw_each_dim, input_each_dim, basis_transformed, x_labels):
    print("b3", b3.shape)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(b3)
    ax[0].set_title('b3')   # separate basis for each feature
    ax[1].plot(b_multi)
    ax[1].set_title('b_multi')
    plt.tight_layout()
    plt.show()
    plt.close()

    for _ in range(5):
        random_session = np.random.choice(len(inputs_raw))
        idxs = np.random.choice(inputs_raw[random_session].shape[0], 10)
        fig, ax = plt.subplots(5, 1, figsize=(18, 10))

        print(inputs_raw[random_session][idxs].T.shape)
        miny = np.min(inputs_raw[random_session][idxs])
        ax[0].plot(inputs_raw[random_session][idxs].T, '-')
        ax[0].set_title(f'Raw input series ({basis_transformed})')

        print(inputs[random_session][idxs].T.shape)
        ax[1].plot(inputs[random_session][idxs].T, '.-')
        ax[1].set_title(f'Basis transformed series ({basis_transformed}) (b3)')

        inpr_invtr = np.array([np.array([BasisProjection(b3).inverse_transform(_) for _ in inpr.reshape(-1, input_each_dim)]).reshape(-1) for inpr in inputs[random_session][idxs]])
        ax[2].plot(inpr_invtr.T, '-')
        ax[2].set_title(f'Basis inverse-transformed series ({basis_transformed}) (b3)')

        print(inputs_multi[random_session][idxs].T.shape)
        ax[3].plot(inputs_multi[random_session][idxs].T, '.-')
        ax[3].set_title(f'Basis transformed series ({basis_transformed}) (bmulti)')

        ax[4].plot(BasisProjection(b_multi).inverse_transform(inputs_multi[random_session][idxs]).T, '-')
        ax[4].set_title(f'Basis inverse-transformed series ({basis_transformed}) (bmulti)')

        # plot vertical lines
        c = 0
        for _ in x_labels:
            ax[0].axvline(c * input_raw_each_dim, ls=':', c='k')
            ax[1].axvline(c * input_each_dim, ls=':', c='k')
            ax[2].axvline(c * input_raw_each_dim, ls=':', c='k')
            ax[3].axvline(c * input_each_dim, ls=':', c='k')
            ax[4].axvline(c * input_raw_each_dim, ls=':', c='k')
            ax[0].text(c * input_raw_each_dim + 1, miny, _, color='r', rotation=90)
            c += 1

        plt.tight_layout()
        plt.show()
        plt.close()
    return


def get_x_and_y_data(datacls, sessions_features, config, display=False):
    """
    Memory-efficient version that saves each session individually.

    This version:
    1. Processes each session one at a time
    2. Saves each session to disk immediately
    3. Only keeps metadata in memory
    4. Returns a data structure with references to session files
    """

    basis_transformed = config['basis_transformed']
    x_labels = config['input_labels_list']
    ax_labels = config['auxiliary_input_labels_list']
    a_labels = config['auxiliary_labels_list']
    y_labels = config['emission_labels']
    ay_labels = config['auxiliary_emission_labels']
    n_inputs = len(x_labels)
    emission_dim = len(y_labels)
    input_raw_each_dim = config['input_raw_each_dim']
    predict_window_size = config['predict_window_size']
    input_raw_overlap = config['input_raw_overlap']
    predict_gap_size = config['predict_gap_size']

    # Cosine basis transformation of inputs
    input_each_dim = config['ncos']
    b = multifeature_basis(raised_cosine(0, input_each_dim, [0, 3 * input_raw_each_dim / 4], 10, input_raw_each_dim), 1)
    basis = scipy.linalg.orth(multifeature_basis(b, 1))
    print("Basis created.")

    # Create temporary directory for session files
    temp_dir = Path('../data/temp_sessions')
    temp_dir.mkdir(exist_ok=True)
    print(f"Temporary session files will be saved to: {temp_dir}")

    # Metadata to collect
    session_keys = []
    copulation_bools = []
    output_mn_std_list = []
    aux_mn_std_list = []
    auxem_mn_std_list = []
    start_frames_list = []
    end_frames_list = []
    num_sessions = 0

    # PHASE 1: Process each session and save immediately
    print("\n=== PHASE 1: Processing and saving sessions ===")
    for s_i, s in enumerate(sessions_features):
        if s_i == 0:
            print('Available features:', sessions_features[s].keys())

        session_len = len(sessions_features[s]['mFV'])
        print(f"Session {s_i} ({s}) length: {session_len}.")

        num_timesteps = session_len
        print(f"Proceeding with session {s_i}.")

        session_copulation = datacls.get_copulation_bool_from_session(s, session_len)
        print(f'Session {s_i} copulation={session_copulation} session_len={session_len}.')

        input_windows, output_windows = create_x_and_y_windows(
            num_timesteps,
            x_size=input_raw_each_dim,
            y_size=predict_window_size,
            x_overlap=input_raw_overlap,
            y_gap_size=predict_gap_size
        )

        s_start_frame = (session_len - num_timesteps)
        s_input_windows = input_windows + s_start_frame
        s_output_windows = output_windows + s_start_frame
        s_end_frame = s_output_windows[-1, 0]
        s_downsampled_indices = s_output_windows[:, 0]
        s_upsampled_indices = np.repeat(np.arange(len(s_downsampled_indices)), predict_window_size)

        # INPUTS
        feats = []
        for label in x_labels:
            f_ = get_input_feat(sessions_features, s, label)
            f = f_[s_input_windows]
            feats.append(f)
        s_inputs_raw = np.hstack(feats)
        del feats
        gc.collect()
        print(f"session {s_i} inp processed")

        # EMISSIONS
        o_feats = []
        o_mn_std = []
        for label in y_labels:
            f, mn, std = get_output_feat(sessions_features, s, label, s_output_windows)
            o_feats.append(f)
            o_mn_std.append([mn, std])
        s_emissions = np.vstack(o_feats).T
        s_o_mn_std = np.vstack(o_mn_std)
        del o_feats
        gc.collect()
        print(f"session {s_i} output processed")

        # AUXILIARY INPUTS
        feats = []
        for label in ax_labels:
            f_ = get_input_feat(sessions_features, s, label)
            f = f_[s_input_windows]
            feats.append(f)
        s_auxem_inputs_raw = np.hstack(feats)
        del feats
        gc.collect()
        print(f"session {s_i} aux inp processed")

        # AUXILIARY EMISSIONS
        ay_feats = []
        ay_mn_std = []
        for label in ay_labels:
            f, mn, std = get_output_feat(sessions_features, s, label, s_output_windows)
            ay_feats.append(f)
            ay_mn_std.append([mn, std])
        s_aux_emissions = np.vstack(ay_feats).T
        s_ay_mn_std = np.vstack(ay_mn_std)
        del ay_feats
        gc.collect()
        print(f"session {s_i} auxem processed")

        # AUXILIARY DATA
        a_feats = []
        a_mn_std = []
        for label in a_labels:
            f, mn, std = get_aux_feat(sessions_features, s, label, s_output_windows)
            a_feats.append(f)
            a_mn_std.append([mn, std])
        s_aux_data = np.vstack(a_feats).T
        s_a_mn_std = np.vstack(a_mn_std)
        del a_feats
        gc.collect()
        print(f"session {s_i} aux processed")

        # Save this session's raw data
        session_data = {
            'inputs_raw': s_inputs_raw,
            'auxem_inputs_raw': s_auxem_inputs_raw,
            'emissions': s_emissions,
            'aux_data': s_aux_data,
            'aux_emissions': s_aux_emissions,
            'downsampled_indices': s_downsampled_indices,
            'upsampled_indices': s_upsampled_indices,
        }

        session_file = temp_dir / f'session_{s_i:04d}.pkl'
        joblib.dump(session_data, session_file)
        print(f"Saved session {s_i} to {session_file}")

        # Collect metadata (lightweight)
        session_keys.append(s)
        copulation_bools.append(session_copulation)
        output_mn_std_list.append(s_o_mn_std)
        aux_mn_std_list.append(s_a_mn_std)
        auxem_mn_std_list.append(s_ay_mn_std)
        start_frames_list.append(s_start_frame)
        end_frames_list.append(s_end_frame)

        # Clear large objects from memory
        del s_inputs_raw, s_auxem_inputs_raw, s_emissions, s_aux_data, s_aux_emissions
        del session_data
        gc.collect()

        num_sessions += 1
        print(f"Completed session {num_sessions}")
        print("============")

        # if num_sessions == 25:
        #     break

    print(f"\nPhase 1 complete: {num_sessions} sessions saved to disk")

    # PHASE 2: Basis transformation (load one at a time)
    print("\n=== PHASE 2: Applying basis transformation ===")
    for s_i in range(num_sessions):
        print(f"Transforming session {s_i}...")
        session_file = temp_dir / f'session_{s_i:04d}.pkl'
        session_data = joblib.load(session_file)

        # Transform inputs
        inputs_raw = session_data['inputs_raw']
        inputs_transformed = transform_single_session(s_i, inputs_raw, basis, input_raw_each_dim)
        session_data['inputs'] = inputs_transformed
        del session_data['inputs_raw']
        del inputs_raw, inputs_transformed
        gc.collect()

        # Transform auxiliary inputs
        auxem_inputs_raw = session_data['auxem_inputs_raw']
        auxem_inputs_transformed = transform_single_session(s_i, auxem_inputs_raw, basis, input_raw_each_dim)
        session_data['auxem_inputs'] = auxem_inputs_transformed
        del session_data['auxem_inputs_raw']
        del auxem_inputs_raw, auxem_inputs_transformed
        gc.collect()

        # Save back with transformations
        joblib.dump(session_data, session_file)
        del session_data
        gc.collect()
        print(f"Session {s_i} transformed and saved")

    print("\nPhase 2 complete: All sessions transformed")

    # Get dimensions from first session
    first_session = joblib.load(temp_dir / 'session_0000.pkl')
    input_dim = first_session['inputs'].shape[-1]
    del first_session
    gc.collect()

    # Create input masks
    n_inputs_by_emission = [len(config['emission_labels'][o]) for o in config['emission_labels']]
    input_sizes_by_emission = [_ * input_each_dim for _ in n_inputs_by_emission]
    blocks = [np.ones(s) for s in input_sizes_by_emission]
    input_mask_by_emission = block_diag(*blocks).astype(int)
    print(n_inputs_by_emission, input_sizes_by_emission)
    print("input_mask_by_emission", input_mask_by_emission)

    n_inputs_by_auxemission = [len(config['auxiliary_emission_labels'][o]) for o in config['auxiliary_emission_labels']]
    input_sizes_by_auxemission = [_ * input_each_dim for _ in n_inputs_by_auxemission]
    blocks = [np.ones(s) for s in input_sizes_by_auxemission]
    input_mask_by_auxemission = block_diag(*blocks).astype(int)
    print(n_inputs_by_auxemission, input_sizes_by_auxemission)
    print("input_mask_by_auxemission", input_mask_by_auxemission)

    # Update config
    config['input_dim'] = input_dim
    config['emission_dim'] = emission_dim
    config['input_each_dim'] = input_each_dim
    config['n_inputs'] = n_inputs
    config['basis'] = basis
    config['input_mask_by_emission'] = input_mask_by_emission
    config['input_mask_by_auxemission'] = input_mask_by_auxemission
    config['num_sessions'] = num_sessions
    config['session_keys'] = session_keys
    config['temp_session_dir'] = str(temp_dir)

    # Create final data structure (metadata only)
    data = {
        'num_sessions': num_sessions,
        'temp_session_dir': str(temp_dir),
        'output_mn_std': np.array(output_mn_std_list),
        'aux_mn_std': np.array(aux_mn_std_list),
        'auxem_mn_std': np.array(auxem_mn_std_list),
        'start_frames': np.array(start_frames_list),
        'end_frames': np.array(end_frames_list),
        'data_config': config,
        'copulation_bools': np.array(copulation_bools),
    }

    print(f"\n=== Processing Complete ===")
    print(f"Session files stored in: {temp_dir}")
    print(f"Total sessions: {num_sessions}")
    print(f"\nTo load a specific session:")
    print(f"  session_data = joblib.load('{temp_dir}/session_XXXX.pkl')")
    print(f"\nTo load all sessions (if you have RAM):")
    print(f"  all_data = load_all_sessions_into_memory(data)")

    return data


def load_all_sessions_into_memory(data):
    """
    Load all session data into memory in the original format.
    Only use this if you have sufficient RAM!
    """
    print(data.keys())
    num_sessions = data['num_sessions']
    temp_dir = Path(data['temp_session_dir'])

    print(f"Loading {num_sessions} sessions into memory...")

    emissions = []
    inputs = []
    auxem_inputs = []
    aux_data = []
    aux_emissions = []
    downsampled_indices = []
    upsampled_indices = []

    for s_i in range(num_sessions):
        session_data = joblib.load(temp_dir / f'session_{s_i:04d}.pkl')
        emissions.append(session_data['emissions'])
        inputs.append(session_data['inputs'])
        auxem_inputs.append(session_data['auxem_inputs'])
        aux_data.append(session_data['aux_data'])
        aux_emissions.append(session_data['aux_emissions'])
        downsampled_indices.append(session_data['downsampled_indices'])
        upsampled_indices.append(session_data['upsampled_indices'])

        if (s_i + 1) % 10 == 0:
            print(f"Loaded {s_i + 1}/{num_sessions} sessions...")

    # Convert to object arrays (original format)
    full_data = {
        'emissions': np.array(emissions, dtype=object),
        'inputs': np.array(inputs, dtype=object),
        'auxem_inputs': np.array(auxem_inputs, dtype=object),
        'aux_data': np.array(aux_data, dtype=object),
        'aux_emissions': np.array(aux_emissions, dtype=object),
        'downsampled_indices': np.array(downsampled_indices, dtype=object),
        'upsampled_indices': np.array(upsampled_indices, dtype=object),
    }

    # Add metadata
    full_data.update(data)

    print("All sessions loaded into memory")
    return full_data


def load_single_session(data, session_idx):
    """Load a specific session's data"""
    temp_dir = Path(data['temp_session_dir'])
    session_file = temp_dir / f'session_{session_idx:04d}.pkl'
    return joblib.load(session_file)


def extract_female(source):

    data_config = {}

    if source == 'wt':
        sessions_features = joblib.load('../data/wt/sessions_features_74_sep5.pkl')
        datacls = WT_DATA
    elif source == 'ac_both':
        sessions_features = joblib.load('../data/ac_both/sessions_features_21_may9.pkl')
        datacls = AC_BOTH
    elif source == 'wt_fred':
        sessions_features = joblib.load('../data/wt_fredcleaned/sessions_features_11_sep5.pkl')
        datacls = FREDCLEANED_DATA
    else:
        raise Exception('Wrong data source.')

    fps = sessions_features.get('fps', datacls.fps)
    data_config['source'] = source
    data_config['orig_fps'] = fps
    data_config['input_raw_each_dim'] = 3*fps
    data_config['predict_gap_size'] = 0     # any gap between x inputs and y output
    data_config['input_raw_overlap'] = fps//30    # move input window forward by 33ms (TODO keep this same as predict window size?)
    data_config["predict_window_size"] = fps//30  # averaging emission over this window size
    data_config['effective_fps'] = data_config['orig_fps'] / data_config["predict_window_size"]
    data_config['basis_transformed'] = 'cos'  # 'cos', 'smooth', or 'identity'
    data_config['ncos'] = 4

    data_config['emission_labels'] = OrderedDict({
        'fFV': ['mFV', 'mLS', 'mfDist', 'fmAng_cos', 'pulse_i', 'sine_i', 'tap2'],
        'fLV': ['mFV_directedlr2', 'mLS_directedlr2', 'mfDist_directedlr2', 'wingAlign_song_i_directedlr2', 'pulse_i_directedlr2', 'sine_i_directedlr2', 'tap2_directedlr2'],
        'fAV': ['mFV_directedlr2', 'mLS_directedlr2', 'mfDist_directedlr2', 'wingAlign_song_i_directedlr2', 'pulse_i_directedlr2', 'sine_i_directedlr2', 'tap2_directedlr2'],
    })
    data_config['input_labels_list'] = [data_config['emission_labels'][o] for o in data_config['emission_labels']]
    data_config['input_labels_list'] = [__ for _ in data_config['input_labels_list'] for __ in _]     # unroll
    print(data_config['input_labels_list'])

    data_config['auxiliary_labels_list'] = ['mFV', 'mLS', 'mfDist', 'fmAng_cos', 'fmAng_sin', 'wingAlign', 'pulse_i', 'sine_i', 'tap2', ]  # we basically need full series as well as windowed-versions of inputs
    data_config['auxiliary_emission_labels'] = OrderedDict({
        'wingFlickBin': ['mFV', 'mLS', 'mfDist', 'fmAng_cos', 'pulse_i', 'sine_i', 'tap2', ]
    })
    data_config['auxiliary_input_labels_list'] = [data_config['auxiliary_emission_labels'][o] for o in data_config['auxiliary_emission_labels']]
    data_config['auxiliary_input_labels_list'] = [__ for _ in data_config['auxiliary_input_labels_list'] for __ in _]  # unroll
    print(data_config['auxiliary_input_labels_list'])

    data_config['statetrans_input_labels_list'] = ['mFV', 'mLS', 'mfDist', 'fmAng_cos', 'pulse_i', 'sine_i', 'tap2']

    filename = f'{source}_fly_data_{data_config["basis_transformed"]}={data_config["ncos"]}_ortho_' \
               f'o={data_config["predict_window_size"]}_smoothed_stdset_auxem_1114.pkl'
    s = time.time()
    data = get_x_and_y_data(datacls, sessions_features, data_config, display=False)
    print("Saving at:", filename)
    joblib.dump(data, f'../data/{filename}')
    print("Saved at:", filename)
    print(f"Done in {time.time() - s} seconds.")
    return


if __name__ == '__main__':
    src = 'wt'
    # extract_female(src)
    # extract_male(src)

    # data = joblib.load('../data/wt_fly_data_cos=4_ortho_o=5_smoothed_stdset_auxem_1114_metadata.pkl')
    # full_data = load_all_sessions_into_memory(data)
    # joblib.dump(full_data, '../data/wt_fly_data_cos=4_ortho_o=5_smoothed_stdset_auxem_1114.pkl')

    data = joblib.load('../data/wt_fly_data_cos=4_ortho_o=5_smoothed_stdset_auxem_113.pkl')
    print(data.keys())
    print(data['data_config'].keys())
    data_config = data['data_config']
    data_config['statetrans_input_labels_list'] = ['mFV', 'mLS', 'mfDist', 'fmAng_cos', 'pulse_i', 'sine_i', 'tap2']
    print(data['data_config'].keys())
    joblib.dump(data, '../data/wt_fly_data_cos=4_ortho_o=5_smoothed_stdset_auxem_1114.pkl')



