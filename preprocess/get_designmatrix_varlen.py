import time

import joblib
import numpy as np
from scipy.stats import zscore
import scipy
from collections import OrderedDict
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import scipy.ndimage
import pywt

from glm_utils.preprocessing import BasisProjection
from glm_utils.bases import identity, raised_cosine, multifeature_basis
import matplotlib.pyplot as plt

from leaprig import WT_DATA, AC_BOTH
from new16mic import FREDCLEANED_DATA


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
    elif f_name in ['song', 'sine_i', 'pfast_i', 'song_i', 'tap', 'tap2']:

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
    elif f_name in ['wingAlign']:
        ts = np.min([sf['wingLAristaLAlignAng'],
                       sf['wingRAristaRAlignAng'],
                       sf['wingLAristaRAlignAng'],
                       sf['wingRAristaLAlignAng']], axis=0)
        ts = smooth_gaussian(ts, sigma=3)
        feat = zscore(ts)     # it is okay to zscore these alignment angles as if they are being treated linearly
    elif f_name in ['song_directed', 'sine_i_directed', 'pfast_i_directed', 'song_i_directed', 'tap_directed', 'tap2_directed']:
        f_name_ = f_name.split('_directed')[0]
        feat = sf[f_name_] * np.sign(np.sin(np.radians(sf['fmAng'])))

        # fig, ax = plt.subplots(2, 1, figsize=(20, 6), sharex=True)
        # ax[0].plot(feat, label=f'{f_name}')
        # ax[1].plot(safe_zscore(feat), label=f'z-{f_name}')
        # ax[0].legend()
        # ax[1].legend()
        # plt.suptitle(s)
        # plt.tight_layout()
        # plt.show()

        feat = safe_zscore(feat)
    else:
        raise Exception(f'unsupported {f_name} input feature.')
    # print(f_name, "done.")
    return feat


def get_aux_feat(sessions_features, s, f_name, aux_windows):
    sf = sessions_features[s]
    if f_name in ['mFV', 'mFS', 'mLS', 'mLV', 'mFA', 'mfDist', 'fFV', 'fFS', 'fLS', 'fLV']:
        ts = sf[f_name]
        ts = smooth_gaussian(ts, sigma=3)
        mn = ts.mean()
        std = ts.std()
        feat = np.mean(zscore(ts)[aux_windows], axis=1)
    elif f_name in ['pfast_i', 'sine_i', 'tap', 'tap2']:
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
    else:
        raise Exception(f'unsupported {f_name} aux feature.')
    return feat, mn, std


def wavelet_denoise(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    print(len(coeffs), [len(_) for _ in coeffs])
    threshold = 0.25 * np.max(coeffs[-1])  # Set small wavelet coefficients to zero
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(coeffs, 'db4')
    return denoised_signal


def get_output_feat(sessions_features, s, f_name, output_windows):
    sf = sessions_features[s]
    if f_name in ['fFV', 'fFS', 'fLS', 'fLV', 'fFA', 'mFV', 'mFS', 'mLS', 'mLV']:
        ts = sf[f_name]
        ts = smooth_gaussian(ts, sigma=3)
        mn = ts.mean()
        std = ts.std()
        f = np.mean(zscore(ts)[output_windows], axis=1)
    elif f_name in ['dfTheta']:
        ts = sf['fTheta']
        ts = smooth_gaussian(ts, sigma=3)
        fTheta = ts[output_windows]
        dfTheta = fTheta[:, -1] - fTheta[:, 0]
        dfTheta = np.where(np.abs(dfTheta) > 90, 0, dfTheta)
        mn = dfTheta.mean()
        std = dfTheta.std()
        dfTheta = zscore(dfTheta)
        f = dfTheta
    elif f_name in ['dfTheta_abs']:
        ts = sf['fTheta']
        ts = smooth_gaussian(ts, sigma=3)
        fTheta = ts[output_windows]
        dfTheta_abs = np.abs(fTheta[:, -1] - fTheta[:, 0])
        dfTheta_abs = np.where(dfTheta_abs > 90, 0, dfTheta_abs)
        mn = dfTheta_abs.mean()
        std = dfTheta_abs.std()
        dfTheta_abs = zscore(dfTheta_abs)
        f = dfTheta_abs
    elif f_name in ['dmTheta']:
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
        # r = np.r_[:100000]
        #
        # fig, ax = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
        #
        # ax[0].plot(np.diff(np.abs(sf['fmAng'][r])), label='dfmAng_abs')
        # ax[0].plot(np.diff(np.abs(sf['fTheta'][r])), label='dfTheta_abs')
        #
        # ax[1].plot(uniform_filter1d(np.diff(np.abs(sf['fmAng'][r])), size=10), label='dfmAng_abs_smoothed')
        # ax[1].plot(uniform_filter1d(np.diff(np.abs(sf['fTheta'][r])), size=10), label='dfTheta_abs_smoothed')
        #
        # ax[2].plot(np.diff(np.abs(sf['fmAng'][r])), label='dfmAng_abs_wavedec')
        # ax[2].plot(np.diff(np.abs(sf['fTheta'][r])), label='dfTheta_abs_wavedec')
        #
        # plt.suptitle('fmAng_abs vs dfTheta_abs')
        # ax[0].legend(loc='upper left')
        # ax[1].legend(loc='upper left')
        # ax[2].legend(loc='upper left')
        # plt.tight_layout()
        # plt.show()
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
    elif f_name in ['wingFlickTheta']:
        ts = sf.get('wingFlickAngle', sf.get('wingMaxAngle'))
        ts = smooth_gaussian(ts, sigma=3)
        wingFlickTheta = ts * sf['wingFlick']
        wingFlickTheta = np.mean(wingFlickTheta[output_windows], axis=1)    # mean can be taken for these angles as they are bounded between 10 and 30 degrees
        mn = 0  # no zscoring for wing flick angles, as most of them are zeros
        std = 1
        f = wingFlickTheta
    elif f_name in ['wingFlickBin']:
        ts = sf['wingFlick']
        wingFlickBin = (np.sum(ts[output_windows], axis=1) >= 1).astype(float)
        mn = 0
        std = 1
        f = wingFlickBin
        # print(np.unique(f, return_counts=True))

        # print(sf.keys())
        # fig = plt.figure(figsize=(20, 4))
        # ax = plt.gca()
        #
        # # Find where it goes from 0→1 and 1→0
        # diff = np.diff(ts, prepend=0, append=0)
        # starts = np.where(diff == 1)[0]
        # ends = np.where(diff == -1)[0]
        #
        # # Shade regions where binary == 1
        # for start, end in zip(starts, ends):
        #     ax.axvspan(start, end, color='orange', alpha=0.3)
        #
        # ax.plot(sf['wingFL'], label='wingFL')
        # ax.plot(sf['wingFR'], label='wingFR')
        # plt.suptitle(s)
        # plt.tight_layout()
        # plt.show()

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

        ax[0].plot(inputs_raw[random_session][idxs].T)
        ax[0].set_title(f'Raw input series ({basis_transformed})')

        ax[1].plot(inputs[random_session][idxs].T)
        ax[1].set_title(f'Basis transformed series ({basis_transformed})')

        ax[2].plot(BasisProjection(basis).inverse_transform(inputs[random_session][idxs]).T)
        ax[2].set_title(f'Basis inverse-transformed series ({basis_transformed})')

        # plot vertical lines
        c = 0
        for _ in x_labels:
            ax[0].axvline(c * input_raw_each_dim, ls=':', c='k')
            ax[1].axvline(c * input_each_dim, ls=':', c='k')
            ax[2].axvline(c * input_raw_each_dim, ls=':', c='k')
            ax[0].text(c * input_raw_each_dim + 1, 0.1, x_labels[_], color='r', rotation=90)
            c += 1

        plt.tight_layout()
        plt.show()
        plt.close()
    return


def get_x_and_y_data(datacls, sessions_features, config, display=False):

    basis_transformed = config['basis_transformed']

    x_labels = config['input_labels']
    y_labels = config['emission_labels']
    a_labels = config['auxiliary_labels']
    ay_labels = config['auxiliary_emission_labels']
    n_inputs = len(x_labels)
    emission_dim = len(y_labels)
    input_raw_each_dim = config['input_raw_each_dim']
    input_raw_dim = input_raw_each_dim * n_inputs
    predict_window_size = config['predict_window_size']
    input_raw_overlap = config['input_raw_overlap']
    predict_gap_size = config['predict_gap_size']

    # Cosine basis transformation of inputs
    input_each_dim = config['ncos']
    b = raised_cosine(0, input_each_dim, [0, 3*input_raw_each_dim/4], 10, input_raw_each_dim)
    b_multi = multifeature_basis(b, n_inputs)
    basis_ortho = scipy.linalg.orth(b_multi)
    basis = basis_ortho
    print("Basis created.")

    inputs_raw = []
    emissions = []
    aux_data = []
    aux_emissions = []
    output_mn_std = []
    aux_mn_std = []
    auxem_mn_std = []
    start_frames = []
    end_frames = []
    downsampled_indices = []
    upsampled_indices = []
    session_keys = []
    copulation_bools = []
    num_sessions = 0

    for s_i, s in enumerate(sessions_features):
        if s_i == 0:
            print('Available features:', sessions_features[s].keys())
        if s_i == 41:
            continue
        session_len = len(sessions_features[s]['mFV'])
        print(f"Session {s_i} ({s}) length: {session_len}.")

        num_timesteps = session_len
        # if session_len < num_timesteps:
        #     print(f"Too short. Skipped.\n============")
        #     continue

        print(f"Proceeding with the session {s_i}.")
        session_keys.append(s)
        session_copulation = datacls.get_copulation_bool_from_session(s, session_len)
        print(f'Session {s_i} copulation={session_copulation} session_len={session_len}.')

        input_windows, output_windows = create_x_and_y_windows(num_timesteps, x_size=input_raw_each_dim,
                                                               y_size=predict_window_size, x_overlap=input_raw_overlap,
                                                               y_gap_size=predict_gap_size)

        s_start_frame = (session_len-num_timesteps)     # index of the first frame of this session used for modeling
        s_input_windows = input_windows + s_start_frame
        s_output_windows = output_windows + s_start_frame
        s_end_frame = s_output_windows[-1, 0]           # index of the last frame of this session used for modeling
        s_downsampled_indices = s_output_windows[:, 0]
        s_upsampled_indices = np.repeat(np.arange(len(s_downsampled_indices)), predict_window_size)

        # INPUTS
        feats = []
        for _ in x_labels:
            f_ = get_input_feat(sessions_features, s, _)
            # print(_, "mean=", np.mean(f_), "std", np.std(f_))
            f = f_[s_input_windows]
            feats.append(f)
        print(f"session {s_i} inp computed")
        s_inputs = np.hstack(feats)
        print(f"session {s_i} inp processed")

        # EMISSIONS
        o_feats = []
        o_mn_std = []
        for _ in y_labels:
            f, mn, std = get_output_feat(sessions_features, s, _, s_output_windows)
            # print(_, f.shape)
            o_feats.append(f)
            o_mn_std.append([mn, std])
        s_emissions = np.vstack(o_feats).T
        s_o_mn_std = np.vstack(o_mn_std)
        print(f"session {s_i} output processed")

        # AUXILIARY EMISSIONS
        ay_feats = []
        ay_mn_std = []
        for _ in ay_labels:
            f, mn, std = get_output_feat(sessions_features, s, _, s_output_windows)
            ay_feats.append(f)
            ay_mn_std.append([mn, std])
        s_aux_emissions = np.vstack(ay_feats).T
        s_ay_mn_std = np.vstack(ay_mn_std)
        print(f"session {s_i} auxem processed")

        # AUXILIARY DAta
        a_feats = []
        a_mn_std = []
        for _ in a_labels:
            f, mn, std = get_aux_feat(sessions_features, s, _, s_output_windows)   # aux windows are the same as output windows since we want to be able to compare outputs and aux data on the same timescale
            a_feats.append(f)
            a_mn_std.append([mn, std])
        s_aux_data = np.vstack(a_feats).T
        s_a_mn_std = np.vstack(a_mn_std)
        print(f"session {s_i} aux processed")
        print(f"session {s_i} processed")

        inputs_raw.append(s_inputs)
        emissions.append(s_emissions)
        aux_data.append(s_aux_data)
        aux_emissions.append(s_aux_emissions)
        output_mn_std.append(s_o_mn_std)
        aux_mn_std.append(s_a_mn_std)
        auxem_mn_std.append(s_ay_mn_std)
        start_frames.append(s_start_frame)
        end_frames.append(s_end_frame)
        downsampled_indices.append(s_downsampled_indices)
        upsampled_indices.append(s_upsampled_indices)
        copulation_bools.append(session_copulation)
        num_sessions += 1
        print("num_sessions", num_sessions)
        print("============")
        # if num_sessions == 5:
        #     break

    inputs_raw = np.array(inputs_raw, dtype=object)
    emissions = np.array(emissions, dtype=object)
    aux_data = np.array(aux_data, dtype=object)
    aux_emissions = np.array(aux_emissions, dtype=object)
    downsampled_indices = np.array(downsampled_indices, dtype=object)
    upsampled_indices = np.array(upsampled_indices, dtype=object)
    output_mn_std = np.array(output_mn_std)
    aux_mn_std = np.array(aux_mn_std)
    auxem_mn_std = np.array(auxem_mn_std)

    print("Basis transforming now..")
    print(len(inputs_raw))
    inputs_transformed = [BasisProjection(basis).transform(_) for _ in inputs_raw]

    print("Basis transformed.")
    inputs = np.array(inputs_transformed, dtype=object)
    input_dim = inputs[0].shape[-1]

    print("basis", basis.shape, "input_raw_each_dim", input_raw_each_dim, "input_raw_dim", input_raw_dim,
          "input_dim", input_dim, "input_each_dim", input_each_dim)
    print("inputs.shape, inputs_raw.shape, emissions.shape, aux_data.shape, aux_emissions.shape",
          inputs.shape, inputs_raw.shape, emissions.shape, aux_data.shape, aux_emissions.shape)
    print("inputs[0].shape, inputs_raw[0].shape, emissions[0].shape, aux_data[0].shape, aux_emissions[0].shape",
          inputs[0].shape, inputs_raw[0].shape, emissions[0].shape, aux_data[0].shape, aux_emissions[0].shape)

    # emissions = np.array(emissions)
    # aux_data = np.array(aux_data)
    # downsampled_indices = np.array(downsampled_indices)
    # upsampled_indices = np.array(upsampled_indices)
    # output_mn_std = np.array(output_mn_std)

    # enhance the config
    config['input_dim'] = input_dim
    config['emission_dim'] = emission_dim
    config['input_each_dim'] = input_each_dim
    config['n_inputs'] = n_inputs
    config['basis'] = basis
    config['num_sessions'] = num_sessions   # number of total sessions
    config['session_keys'] = session_keys   # sessions in this data in order

    data = {
        'emissions': emissions,
        'inputs': inputs,
        'aux_data': aux_data,
        'aux_emissions': aux_emissions,
        'output_mn_std': np.array(output_mn_std),
        'aux_mn_std': np.array(aux_mn_std),
        'auxem_mn_std': np.array(auxem_mn_std),
        'start_frames': np.array(start_frames),
        'end_frames': np.array(end_frames),
        'downsampled_indices': downsampled_indices,
        'upsampled_indices': upsampled_indices,
        'data_config': config,
        'copulation_bools': np.array(copulation_bools),
    }

    # Plot few samples of inputs
    if display:
        some_plots(b_multi, basis_ortho, inputs_raw, inputs, input_raw_each_dim, input_each_dim, basis, basis_transformed, x_labels)

    return data


def extract_male(source):

    data_config = {}

    if source == 'wt':
        sessions_features = joblib.load('../data/wt/sessions_features_75_may30.pkl')
        datacls = WT_DATA
    elif source == 'ac_both':
        sessions_features = joblib.load('../data/ac_both/sessions_features_21_may9.pkl')
        datacls = AC_BOTH
    elif source == 'wt_fred':
        sessions_features = joblib.load('../data/wt_fredcleaned/sessions_features_11_may30.pkl')
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
    data_config['input_labels'] = OrderedDict({
        'fFV': 'z-fFV',
        'fLS': 'z-fLS',
        'mfDist': 'z-mfDist',

        'mfAng_cos': 'front_back',

    })
    data_config['emission_labels'] = OrderedDict({
        'mFV': 'z-mFV',
        'mLV': 'z-mLV',
        'dmTheta': 'z-dmTheta',
        # 'dfmAng': 'z-dfmAng',
    })
    data_config['auxiliary_labels'] = OrderedDict({
        'fFV': 'z-fFV',     # we basically need full series as well as windowed-versions of inputs
        'fLS': 'z-fLS',
        'mfDist': 'z-mfDist',
        'mfAng_cos': 'mfAng_cos',
        # 'fmAng_sin': 'fmAng_sin',
    })
    data_config['auxiliary_emission_labels'] = OrderedDict({
        'wingFlickBin': 'wingFlickBin',
        # 'wingFlickTheta': 'wingAngFlick',
    })

    filename = f'{source}_fly_data_{data_config["basis_transformed"]}={data_config["ncos"]}_ortho_' \
               f'o={data_config["predict_window_size"]}_smoothed_stdset_auxem_MALE.pkl'
    s = time.time()
    data = get_x_and_y_data(datacls, sessions_features, data_config, display=False)
    print("Saving at:", filename)
    joblib.dump(data, f'../data/{filename}')
    print("Saved at:", filename)
    print(f"Done in {time.time() - s} seconds.")
    return


def extract_female(source):

    data_config = {}

    if source == 'wt':
        sessions_features = joblib.load('../data/wt/sessions_features_75_may30.pkl')
        datacls = WT_DATA
    elif source == 'ac_both':
        sessions_features = joblib.load('../data/ac_both/sessions_features_21_may9.pkl')
        datacls = AC_BOTH
    elif source == 'wt_fred':
        sessions_features = joblib.load('../data/wt_fredcleaned/sessions_features_11_may30.pkl')
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
    data_config['input_labels'] = OrderedDict({
        'mFV': 'z-mFV',
        'mLS': 'z-mLS',
        'mfDist': 'z-mfDist',

        # 'fmAng_sin': 'maleLR',
        'fmAng_cos': 'front_back',

        'wingAlign': 'z-wingAlign',
        'pfast_i': 'z-pulse',
        'sine_i': 'z-sine',
        'pfast_i_directed': 'pulseLR',
        'sine_i_directed': 'sineLR',

        'tap2': 'z-tap2',
        'tap2_directed': 'tap2LR',

        # 'fDistWall': 'distWall',
    })
    data_config['emission_labels'] = OrderedDict({
        'fFV': 'z-fFV',
        'fLV': 'z-fLV',
        'dfTheta': 'z-dfTheta',
        # 'dfmAng': 'z-dfmAng',
    })
    data_config['auxiliary_labels'] = OrderedDict({
        'mFV': 'z-mFV',     # we basically need full series as well as windowed-versions of inputs
        'mLS': 'z-mLS',
        'mfDist': 'z-mfDist',
        'pfast_i': 'z-pulse',
        'sine_i': 'z-sine',
        'tap2': 'z-tap2',
        'fmAng_cos': 'fmAng_cos',
        # 'fmAng_sin': 'fmAng_sin',
    })
    data_config['auxiliary_emission_labels'] = OrderedDict({
        'wingFlickBin': 'wingFlickBin',
        # 'wingFlickTheta': 'wingAngFlick',
    })

    filename = f'{source}_fly_data_{data_config["basis_transformed"]}={data_config["ncos"]}_ortho_' \
               f'o={data_config["predict_window_size"]}_smoothed_stdset_auxem.pkl'
    s = time.time()
    data = get_x_and_y_data(datacls, sessions_features, data_config, display=False)
    print("Saving at:", filename)
    joblib.dump(data, f'../data/{filename}')
    print("Saved at:", filename)
    print(f"Done in {time.time() - s} seconds.")
    return


if __name__ == '__main__':
    src = 'wt'
    extract_female(src)
    # extract_male(src)
