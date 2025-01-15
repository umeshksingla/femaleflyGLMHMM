import joblib
import numpy as np
from scipy.stats import zscore
import scipy
from collections import OrderedDict
from scipy.signal import savgol_filter

from utils.utils import create_x_and_y_windows
from glm_utils.preprocessing import BasisProjection
from glm_utils.bases import identity, raised_cosine, multifeature_basis
import matplotlib.pyplot as plt


DATA = 'wt'
# BASE_FOLDER = f' notebooks/eda/{DATA}/'
sessions_features = joblib.load('../data/sessions_features_82_isong.pkl')


def smooth_moving_average(x, smooth_window):
    return np.convolve(x, np.ones(smooth_window), 'valid') / smooth_window


def smooth_savgol(x, smooth_window):
    return savgol_filter(x, window_length=smooth_window, polyorder=1, axis=0)


def get_input_feat(s, f_name):
    if f_name in ['mFV', 'mLS', 'mFA', 'mLA', 'mLV', 'mfDist']:
        return zscore(sessions_features[s][f_name])     # todo: zscore?
    elif f_name in ['song', 'sine_i', 'pfast_i', 'song_i', 'tap']:
        return sessions_features[s][f_name]
    elif f_name in ['fmAng_cos']:
        return np.cos(np.radians(sessions_features[s]['fmAng']))    # cos: front to back
    elif f_name in ['fmAng_sin']:
        return np.sin(np.radians(sessions_features[s]['fmAng']))    # sin: left or right of the fly
    elif f_name in ['song_directed', 'sine_i_directed', 'pfast_i_directed', 'song_i_directed', 'tap_directed']:
        f_name_ = f_name.split('_directed')[0]
        return sessions_features[s][f_name_] * np.sign(np.sin(np.radians(sessions_features[s]['fmAng'])))
    else:
        raise Exception(f'unsupported {f_name} input feature.')


def get_output_feat(s, f_name, output_windows):
    if f_name in ['fFV', 'fFS', 'fLS', 'fLV', 'fFA']:
        ts = sessions_features[s][f_name]
        # ts = smooth_moving_average(ts, 20)
        return np.mean(zscore(ts)[output_windows], axis=1)    # todo: zscore?
    elif f_name in ['dfTheta']:
        fTheta = sessions_features[s]['fTheta'][output_windows]
        dfTheta = fTheta[:, -1] - fTheta[:, 0]
        dfTheta = np.where(np.abs(dfTheta) > 90, 0, dfTheta)
        dfTheta = zscore(dfTheta)
        return dfTheta
    elif f_name in ['dfTheta_abs']:
        fTheta = sessions_features[s]['fTheta'][output_windows]
        dfTheta_abs = np.abs(fTheta[:, -1] - fTheta[:, 0])
        dfTheta_abs = np.where(dfTheta_abs > 90, 0, dfTheta_abs)
        dfTheta_abs = zscore(dfTheta_abs)
        return dfTheta_abs
    else:
        raise Exception(f'unsupported {f_name} output feature.')


def get_x_and_y_data(config, display=False):

    basis_transformed = config['basis_transformed']
    # num_sessions = config['num_sessions']

    x_labels = config['input_labels']
    y_labels = config['emission_labels']
    n_inputs = len(x_labels)
    emission_dim = len(y_labels)
    input_raw_each_dim = config['input_raw_each_dim']
    input_raw_dim = input_raw_each_dim * n_inputs
    num_timesteps = config['num_timesteps']
    predict_window_size = config['predict_window_size']
    input_raw_overlap = config['input_raw_overlap']
    predict_gap_size = config['predict_gap_size']

    inputs_raw = []
    emissions = []
    # extra_data = []

    input_windows, output_windows = create_x_and_y_windows(num_timesteps,
                                                           x_size=input_raw_each_dim,
                                                           y_size=predict_window_size,
                                                           x_overlap=input_raw_overlap,
                                                           y_gap_size=predict_gap_size)

    num_sessions = 0
    session_keys = []
    for s_i, s in enumerate(sessions_features):
        if s_i == 0:
            print(sessions_features[s].keys())
        session_len = len(sessions_features[s]['mFV'])
        if session_len < num_timesteps:    # skip short sessions for now
            print(f"Session {s_i} length = {len(sessions_features[s]['mFV'])}. Too short. Skipped.")
            continue
        # if session_len > 250000:
        #     print(f"Session {s_i} length = {len(sessions_features[s]['mFV'])}. No copulation. Skipped.")
        #     continue
        session_keys.append(s)

        # INPUTS
        feats = []
        for _ in x_labels:
            f = get_input_feat(s, _)[input_windows]
            feats.append(f)
        s_inputs = np.hstack(feats)

        # EMISSIONS
        o_feats = []
        for _ in y_labels:
            f = get_output_feat(s, _, output_windows)
            o_feats.append(f)
        s_emissions = np.vstack(o_feats).T

        inputs_raw.append(s_inputs)
        emissions.append(s_emissions)
        num_sessions += 1

    inputs_raw = np.array(inputs_raw)
    emissions = np.array(emissions)

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

    print("inputs.shape, inputs_raw.shape, emissions.shape", inputs.shape, inputs_raw.shape, emissions.shape)

    # enhance the config
    config['input_dim'] = input_dim
    config['emission_dim'] = emission_dim
    config['input_each_dim'] = input_each_dim
    config['n_inputs'] = n_inputs
    config['basis'] = basis
    config['num_sessions'] = num_sessions   # number of total sessions
    config['session_keys'] = session_keys   # sessions in this data in order

    data = {
            'data_config': config,
            'emissions': emissions,
            'inputs': inputs,
            # 'inputs_raw': inputs_raw,
            'output_indices': output_windows[:, 0],
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

    from standard_data_config import data_config

    data_config['input_raw_each_dim'] = 3*150
    data_config['num_timesteps'] = 100000
    data_config["predict_window_size"] = 5
    data_config['input_labels'] = OrderedDict({
        'mFV': 'z-mFV',
        'mLS': 'z-mLS',
        'fmAng_sin': 'maleLR',

        'pfast_i': 'pulse',
        'sine_i': 'sine',
        'tap': 'tap',

        'pfast_i_directed': 'pulseLR',
        'sine_i_directed': 'sineLR',
        'tap_directed': 'tapLR',

        'mfDist': 'z-mfDist',
    })
    data_config['emission_labels'] = OrderedDict({
        'fFV': 'z-fFV',
        'fLV': 'z-fLV',
        'dfTheta': 'z-fRV',
    })
    data_config['ncos'] = 4

    # describe_sessions()
    # sys.exit()

    data = get_x_and_y_data(data_config, display=True)
    filename = f'fly_data_{data_config["basis_transformed"]}={data_config["ncos"]}_ortho_o={data_config["predict_window_size"]}.pkl'
    print("Saving at:", filename)
    joblib.dump(data, f'../data/{filename}')
    print("Saved at:", filename)
