import joblib
import numpy as np
from scipy.stats import zscore
import scipy
from collections import OrderedDict
from scipy.signal import savgol_filter

from glm_utils.preprocessing import BasisProjection
from glm_utils.bases import identity, raised_cosine, multifeature_basis
import matplotlib.pyplot as plt


def smooth_moving_average(x, smooth_window):
    return np.convolve(x, np.ones(smooth_window), 'valid') / smooth_window


def smooth_savgol(x, smooth_window):
    return savgol_filter(x, window_length=smooth_window, polyorder=1, axis=0)


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


def get_input_feat(s, f_name):
    sf = sessions_features[s]
    if f_name in ['mFV', 'mLS', 'mFA', 'mLA', 'mLV', 'mfDist', 'fDistWall']:
        feat = zscore(sf[f_name])     # todo: zscore?
    elif f_name in ['song', 'sine_i', 'pfast_i', 'song_i', 'tap', 'tap2']:
        feat = sf[f_name].astype(float)
    elif f_name in ['fmAng_cos']:
        feat = np.cos(np.radians(sf['fmAng']))    # cos: front to back
    elif f_name in ['fmAng_sin']:
        feat = np.sin(np.radians(sf['fmAng']))    # sin: left or right of the fly
    elif f_name in ['wingAlign']:
        feat = np.min([sf['wingLAristaLAlignAng'],
                       sf['wingRAristaRAlignAng'],
                       sf['wingLAristaRAlignAng'],
                       sf['wingRAristaLAlignAng']], axis=0)
        feat = zscore(feat)     # it is okay to zscore these alignment angles as they are treated linearly
    elif f_name in ['song_directed', 'sine_i_directed', 'pfast_i_directed', 'song_i_directed', 'tap_directed', 'tap2_directed']:
        f_name_ = f_name.split('_directed')[0]
        feat = sf[f_name_] * np.sign(np.sin(np.radians(sf['fmAng'])))
    else:
        raise Exception(f'unsupported {f_name} input feature.')
    return feat


def get_output_feat(s, f_name, output_windows):
    sf = sessions_features[s]
    if f_name in ['fFV', 'fFS', 'fLS', 'fLV', 'fFA']:
        ts = sf[f_name]
        mn = ts.mean()
        std = ts.std()
        f = np.mean(zscore(ts)[output_windows], axis=1)    # todo: zscore? zscore before mean or after?
    elif f_name in ['dfTheta']:
        fTheta = sf['fTheta'][output_windows]
        dfTheta = fTheta[:, -1] - fTheta[:, 0]
        dfTheta = np.where(np.abs(dfTheta) > 90, 0, dfTheta)
        mn = dfTheta.mean()
        std = dfTheta.std()
        dfTheta = zscore(dfTheta)
        f = dfTheta
    elif f_name in ['dfTheta_abs']:
        fTheta = sf['fTheta'][output_windows]
        dfTheta_abs = np.abs(fTheta[:, -1] - fTheta[:, 0])
        dfTheta_abs = np.where(dfTheta_abs > 90, 0, dfTheta_abs)
        mn = dfTheta_abs.mean()
        std = dfTheta_abs.std()
        dfTheta_abs = zscore(dfTheta_abs)
        f = dfTheta_abs
    else:
        raise Exception(f'unsupported {f_name} output feature.')
    # print(f_name, mn, std)
    return f, mn, std


def get_aux_feat(s, f_name, aux_windows):
    if f_name in ['mFV', 'mFS', 'mLS', 'mLV', 'mFA', 'mfDist']:
        ts = sessions_features[s][f_name]
        # ts = smooth_moving_average(ts, 20)
        feat = np.mean(zscore(ts)[aux_windows], axis=1)    # todo: zscore?
    elif f_name in ['pfast_i', 'sine_i', 'tap', 'tap2', 'tap2_directed']:
        ts = sessions_features[s][f_name]
        feat = np.sum(ts[aux_windows], axis=1)  # todo: zscore?
    else:
        raise Exception(f'unsupported {f_name} aux feature.')
    return feat


def get_x_and_y_data(config, display=False):

    copulation_bool = config['copulation']
    if copulation_bool is True:
        print('Processing sessions with copulation.')
    else:
        print('Processing sessions with no copulation.')

    basis_transformed = config['basis_transformed']
    # num_sessions = config['num_sessions']

    x_labels = config['input_labels']
    y_labels = config['emission_labels']
    a_labels = config['auxiliary_labels']
    n_inputs = len(x_labels)
    emission_dim = len(y_labels)
    input_raw_each_dim = config['input_raw_each_dim']
    input_raw_dim = input_raw_each_dim * n_inputs
    predict_window_size = config['predict_window_size']
    input_raw_overlap = config['input_raw_overlap']
    predict_gap_size = config['predict_gap_size']

    if copulation_bool:
        num_timesteps = 45000
        print("Copulation True")
    else:
        num_timesteps = 270000
        print("Copulation False")

    input_windows, output_windows = create_x_and_y_windows(num_timesteps,
                                                           x_size=input_raw_each_dim,
                                                           y_size=predict_window_size,
                                                           x_overlap=input_raw_overlap,
                                                           y_gap_size=predict_gap_size)

    inputs_raw = []
    emissions = []
    aux_data = []
    output_mn_std = []
    start_frames = []
    end_frames = []
    downsampled_indices = []
    upsampled_indices = []
    session_keys = []
    num_sessions = 0

    for s_i, s in enumerate(sessions_features):
        if s_i == 0:
            print('Available features:', sessions_features[s].keys())
        session_len = len(sessions_features[s]['mFV'])
        print(f"Session {s_i} ({s}) length: {session_len}.")

        session_copulation = session_len < 270000    # TODO: have to verify if sessions > 240k but <270k actually copulate or not
        if copulation_bool != session_copulation:
            print(f'Copulation={session_copulation}. Skipped.')
            continue

        if copulation_bool is True and session_len < num_timesteps:
            print(f"Too short for copulation sessions. Skipped.")
            continue

        session_keys.append(s)
        s_start_frame = (session_len-num_timesteps)     # index of the first frame of this session used for modeling
        s_input_windows = input_windows + s_start_frame
        s_output_windows = output_windows + s_start_frame
        s_end_frame = s_output_windows[-1, 0]           # index of the last frame of this session used for modeling
        s_downsampled_indices = s_output_windows[:, 0]
        s_upsampled_indices = np.repeat(np.arange(len(s_downsampled_indices)), predict_window_size)
        print(f'Session {s_i}', s_start_frame, s_end_frame, "s_downsampled_indices", s_downsampled_indices)
        print(f'Session {s_i}', s_start_frame, s_end_frame, "s_upsampled_indices", s_upsampled_indices)

        # INPUTS
        feats = []
        for _ in x_labels:
            f = get_input_feat(s, _)[s_input_windows]
            feats.append(f)
        s_inputs = np.hstack(feats)

        # EMISSIONS
        o_feats = []
        o_mn_std = []
        for _ in y_labels:
            f, mn, std = get_output_feat(s, _, s_output_windows)
            o_feats.append(f)
            o_mn_std.append([mn, std])
        s_emissions = np.vstack(o_feats).T
        s_o_mn_std = np.vstack(o_mn_std)

        # AUXILIARY DAta
        a_feats = []
        for _ in a_labels:
            f = get_aux_feat(s, _, s_output_windows)   # aux windows are the same as output windows since we want to be able to compare outputs and aux data on the same timescale
            a_feats.append(f)
        s_aux_data = np.vstack(a_feats).T

        inputs_raw.append(s_inputs)
        emissions.append(s_emissions)
        aux_data.append(s_aux_data)
        output_mn_std.append(s_o_mn_std)
        start_frames.append(s_start_frame)
        end_frames.append(s_end_frame)
        downsampled_indices.append(s_downsampled_indices)
        upsampled_indices.append(s_upsampled_indices)
        num_sessions += 1
        print("============")

    inputs_raw = np.array(inputs_raw)
    emissions = np.array(emissions)
    aux_data = np.array(aux_data)   # aux data doesn't need to be basis transformed
    output_mn_std = np.array(output_mn_std)

    # Cosine basis transformation of inputs
    if basis_transformed == 'cos':
        input_each_dim = config['ncos']
        b = raised_cosine(0, input_each_dim, [0, 2*input_raw_each_dim/3], 10, input_raw_each_dim)
        b_multi = multifeature_basis(b, n_inputs)
        # basis = b_multi
        basis_ortho = scipy.linalg.orth(b_multi)
        basis = basis_ortho
        print("basis", basis.shape, "input_raw_each_dim", input_raw_each_dim, "input_raw_dim", input_raw_dim)
        if display:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(b_multi)
            ax[0].set_title('Basis')
            ax[1].plot(basis_ortho)
            ax[1].set_title('Ortho-normalized Basis')
            plt.tight_layout()
            plt.show()
            plt.close()
        inputs = BasisProjection(basis).transform(inputs_raw.reshape(-1, input_raw_dim)).reshape(num_sessions, -1, basis.shape[-1])
        input_dim = inputs.shape[-1]
    elif basis_transformed == 'smooth':
        input_each_dim = 20
        b = raised_cosine(0, input_each_dim, [0, input_raw_each_dim], 1, input_raw_each_dim)  # smoothing using cosine basis
        b_multi = multifeature_basis(b, n_inputs)
        basis = b_multi
        print("basis", basis.shape, "input_raw_each_dim", input_raw_each_dim, "input_raw_dim", input_raw_dim)
        inputs_tr = BasisProjection(basis).transform(inputs_raw.reshape(-1, input_raw_dim))
        inputs = BasisProjection(basis).inverse_transform(inputs_tr).reshape(num_sessions, num_timesteps, -1)
        input_dim = inputs.shape[-1]
    elif basis_transformed == 'identity':
        input_each_dim = input_raw_each_dim
        b = identity(input_raw_each_dim)
        b_multi = multifeature_basis(b, n_inputs)
        basis = b_multi
        print("basis", basis.shape, "input_raw_each_dim", input_raw_each_dim, "input_raw_dim", input_raw_dim)
        inputs = BasisProjection(basis).transform(inputs_raw.reshape(-1, input_raw_dim)).reshape(num_sessions, -1, basis.shape[-1])
        input_dim = inputs.shape[-1]
    else:
        raise Exception('Invalid inputs transformation.')

    print("inputs.shape, inputs_raw.shape, emissions.shape, aux_data", inputs.shape, inputs_raw.shape, emissions.shape, aux_data.shape)

    # enhance the config
    config['input_dim'] = input_dim
    config['emission_dim'] = emission_dim
    config['input_each_dim'] = input_each_dim
    config['n_inputs'] = n_inputs
    config['basis'] = basis
    config['num_sessions'] = num_sessions   # number of total sessions
    config['session_keys'] = session_keys   # sessions in this data in order
    config['num_timesteps'] = num_timesteps

    data = {
        'data_config': config,
        'emissions': emissions,
        'inputs': inputs,
        'aux_data': aux_data,
        'output_mn_std': output_mn_std,
        # 'inputs_raw': inputs_raw,
        'output_indices': output_windows[:, 0], # TODO!!!? need per session output windows now
        'aux_indices': output_windows[:, 0],
        'start_frames': np.array(start_frames),
        'end_frames': np.array(end_frames),
        'downsampled_indices': np.array(downsampled_indices),
        'upsampled_indices': np.array(upsampled_indices),
    }

    # Plot few samples of inputs
    if display:
        idxs = np.random.choice(inputs_raw.shape[1], 10)
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(inputs_raw[0, idxs].T)
        ax[0].set_title(f'Raw input series ({basis_transformed})')

        ax[1].plot(inputs[0, idxs].T)
        ax[1].set_title(f'Basis transformed series ({basis_transformed})')

        if basis_transformed != 'smooth':
            ax[2].plot(
                BasisProjection(basis).inverse_transform(
                    inputs[0, idxs].reshape(-1, inputs.shape[-1])
                ).reshape(len(idxs), -1).T)
            ax[2].set_title(f'Basis inverse-transformed series ({basis_transformed})')

        # plot axvlines
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
    return data


def describe_sessions():
    ns = 0
    for s_i, s in enumerate(sessions_features):
        if len(sessions_features[s]['mFV']) < data_config['num_timesteps']:    # skip short sessions for now
            print(f"Session {s_i} length = {len(sessions_features[s]['mFV'])}. Skipped.")
            continue
        print(s_i, s, "number in data", ns, "Verify no:", len(sessions_features[s]['mFV']))
        ns += 1
    return


if __name__ == '__main__':

    data_config = {}

    source = 'wt'
    cop_ = True
    if source == 'wt':
        sessions_features = joblib.load('../data/wt/sessions_features_77.pkl')
        fps = sessions_features.get('fps', 150)
    elif source == 'wt_fred':
        sessions_features = joblib.load('../data/wt_fredcleaned/sessions_features_11.pkl')
        fps = sessions_features.get('fps', 60)
    else:
        raise Exception('Wrong data source.')

    data_config['source'] = source
    data_config['copulation'] = cop_
    data_config['fps'] = fps
    data_config['input_raw_each_dim'] = 3*fps
    # data_config['num_timesteps'] = 100000
    data_config['predict_gap_size'] = 0     # any gap between x inputs and y output
    data_config['input_raw_overlap'] = fps//30    # move input window forward by 33ms (TODO keep this same as predict window size?)
    data_config["predict_window_size"] = fps//30  # averaging emission over this window size (100ms)
    data_config['input_labels'] = OrderedDict({
        'mFV': 'z-mFV',
        'mLS': 'z-mLS',
        'mfDist': 'z-mfDist',

        'fmAng_sin': 'maleLR',
        'wingAlign': 'z-wingAlign',

        'pfast_i': 'pulse',
        'sine_i': 'sine',
        'tap2': 'tap2',

        'pfast_i_directed': 'pulseLR',
        'sine_i_directed': 'sineLR',
        'tap2_directed': 'tap2LR',

        # 'fDistWall': 'distWall',
    })

    data_config['emission_labels'] = OrderedDict({
        'fFV': 'forward velocity',
        'fLV': 'lateral velocity',
        'dfTheta': 'orientation change',
    })

    data_config['auxiliary_labels'] = OrderedDict({
        'mFV': 'z-mFV',
        'mLS': 'z-mLS',
        'mfDist': 'z-mfDist',
        'pfast_i': 'pulse',
        'sine_i': 'sine',
        'tap2': 'tap2',
    })
    data_config['basis_transformed'] = 'cos'  # 'cos', 'smooth', or 'identity'
    data_config['ncos'] = 4

    # describe_sessions()
    # sys.exit()

    data = get_x_and_y_data(data_config, display=True)
    filename = f'{source}_fly_data_{data_config["basis_transformed"]}={data_config["ncos"]}_ortho_o={data_config["predict_window_size"]}_copulation={data_config["copulation"]}.pkl'
    print("Saving at:", filename)
    joblib.dump(data, f'../data/{filename}')
    print("Saved at:", filename)
