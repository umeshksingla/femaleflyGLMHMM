import os
import glob
import random

import joblib
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
from wonderwords import RandomWord
from datetime import datetime
from collections import defaultdict

from plotting import plots
from utilities.video_utils import clip_session


def getafilepath(model_name):
    foldertime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f'./models/{model_name}/{foldertime}_{RandomWord().word()}'


def load_latest(model_name):
    latest_model_path = sorted(list(glob.glob(f'models/{model_name}/*')), reverse=True)[0]
    return load_specific_path(latest_model_path)


def load_specific_path(model_path):
    print('Loading:', model_path)
    # print(os.listdir(model_path))
    # print(glob.glob(f'{model_path}/*.pkl'))
    data_config_pkl = joblib.load(os.path.join(model_path, 'data_config.pkl'))
    # print(data_config_pkl)
    model_pkl = joblib.load(os.path.join(model_path, 'model.pkl'))
    with open(os.path.join(model_path, 'model_config.json')) as f: model_config = json.load(f)
    with open(os.path.join(model_path, 'SUCCESS.txt')) as f: fit_success = f.read()
    if fit_success != 'True':
        print(Warning(f'Unsuccessful model loaded. {model_path}'))
        return None, None, None
    return model_pkl, data_config_pkl, model_config


def load_specific_path_single(model_path):
    data_config_pkl = joblib.load(os.path.join(model_path, 'data_config.pkl'))
    model_pkl = joblib.load(os.path.join(model_path, 'model_ind.pkl'))
    with open(os.path.join(model_path, 'model_config.json')) as f: model_config = json.load(f)
    return model_pkl, data_config_pkl, model_config


def load_all_singles(model_dir):
    model_pkls = []
    for model_path in glob.glob(model_dir + '/session*'):
        pkl, data_config_pkl, model_config = load_specific_path_single(model_path)
        model_pkls.append(pkl)
    return model_pkls, data_config_pkl, model_config


def calculate_steady_state_p(P):
    """
    Calculates the steady state probabilities of a Markov chain.

    Parameters:
    P (numpy.ndarray): The transition matrix of the Markov chain.

    Returns:
    numpy.ndarray: The steady state probability vector.
    """
    # print(P)
    # Check if the matrix is square
    if P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square.")
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find the eigenvector corresponding to the eigenvalue 1
    index = np.isclose(eigenvalues, 1, rtol=1e-8)
    if not np.any(index):
        raise ValueError("No eigenvalue 1 found.")
    steady_state_vector = eigenvectors[:, index].real
    steady_state_vector = steady_state_vector[:, 0]
    steady_state_vector = steady_state_vector/np.sum(steady_state_vector)
    # print(steady_state_vector)
    return steady_state_vector


def get_emissions_by_state(emissions, stateseq, num_states, output_mn_std=None, rescaled=True):
    emissions_z = {}
    for btch in range(len(stateseq)):
        for z in range(num_states):
            if z not in emissions_z: emissions_z[z] = []
            if rescaled:
                mn_std_btch = output_mn_std[btch]
                eez = emissions[btch][stateseq[btch] == z] * mn_std_btch[:, 1, None].T + mn_std_btch[:, 0, None].T
            else:
                eez = emissions[btch][stateseq[btch] == z]
            emissions_z[z].append(eez)
    for z in emissions_z:
        emissions_z[z] = np.vstack(emissions_z[z])
    return emissions_z


def get_stateseq_indices(indices_seq, state_seq, min_length=10):
    intervals = defaultdict(list)
    transitions_at = np.where(np.diff(state_seq) != 0)[0]+1
    transitions_at = np.insert(transitions_at, 0, 0)
    transitions_at = np.append(transitions_at, len(indices_seq)-1)
    for s, e in zip(transitions_at, transitions_at[1:]):
        if (e-s) >= min_length:
            intervals[state_seq[s]].append((indices_seq[s], indices_seq[e]-1))  # state doesn't end at e-1 frame but original_indexing-1
    return intervals


def normalize_to_equal_length(arr_list, GRID=101):
    resampled = []
    for a in arr_list:
        # print(a.shape)
        T, S = a.shape
        # print("T, S", T, S)
        u_src = np.linspace(0, 1, T)
        u_dst = np.linspace(0, 1, GRID)
        a_rs = np.vstack([np.interp(u_dst, u_src, a[:, s]) for s in range(S)]).T  # (GRID,S)
        resampled.append(a_rs)
    resampled = np.stack(resampled)  # (N, GRID, S)
    # print("resampled", resampled.shape)
    return resampled


def pad_to_equal_length(arr_list):
    # arr_list = [a.reshape(-1, 1) for a in arr_list]
    lengths = np.array([len(a) for a in arr_list])
    max_length = np.max(lengths)
    # sorted_lengths = np.sort(lengths)
    n_le = np.array([np.sum(L <= lengths) for L in np.arange(0, max_length)])
    # print("n_le", n_le)
    padded_arr_list = np.array([np.pad(a, ((0, max_length-a.shape[0]), (0, 0)), constant_values=0) for a in arr_list])

    # print(padded_arr_list.shape)
    return padded_arr_list, n_le


def get_windows_to_plot(effective_fps, num_timestamps):

    # Helper to safely generate window arrays
    def make_windows(window_duration_frames, n_windows):
        max_start = max(0, num_timestamps - window_duration_frames - 10)    # to avoid linspace going negative
        if max_start == 0:
            return np.empty((0, 2), dtype=int)
        starts = np.linspace(0, max_start, num=n_windows)
        starts = np.round(starts).astype(int)
        ends = starts + window_duration_frames
        windows = np.stack((starts, ends), axis=1)
        return windows

    window_size = effective_fps * 1  # 1 second windows
    windows1 = make_windows(window_size * 60, n_windows=10)  # 1 min
    windows2 = make_windows(window_size * 60 * 5, n_windows=5)  # 5 min
    windows3 = make_windows(window_size, n_windows=5)  # 1 sec

    full_session_window = np.array([[0, num_timestamps - 1]])   # Full session window

    windows = np.vstack((windows1, windows2, windows3, full_session_window)).astype(int)

    return windows


def get_cop_window_to_plot(effective_fps, num_timestamps):

    window_size = effective_fps * 1  # 1 second windows
    last_window_size = window_size * 10   # Last 10-second window, to visualize near-copulation
    if num_timestamps > last_window_size:
        last_window = np.array([[num_timestamps - last_window_size, num_timestamps - 1]])
    else:
        last_window = np.empty((0, 2), dtype=int)
    windows = np.vstack((last_window)).astype(int)
    return windows

# def analyze_state_mean(state_seq, config, emissions, inputs):
#     """
#     Inputs and outputs by state
#     :return:
#     """
#     num_batches = len(emissions)
#
#     # get mean cue in each state
#     inputs_z = {}
#     outputs_z = {}
#     for z in range(config['num_states']):
#         inputs_z[z] = []
#         outputs_z[z] = []
#         for b in range(num_batches):
#             z_filter = state_seq[b] == z
#             if np.sum(z_filter) == 0:
#                 print(f'0 samples in this batch for state {z}.')
#                 continue
#             inputs_z[z].append(inputs[b][z_filter])
#             outputs_z[z].append(emissions[b][z_filter])
#         inputs_z[z] = np.vstack(inputs_z[z])
#         outputs_z[z] = np.vstack(outputs_z[z])
#     print(inputs_z[0].shape)
#     print(outputs_z[0].shape)
#     return inputs_z, outputs_z


def save_single(model, emissions, inputs, copulation_bool, trained_bool, output_dir):
    """Save single session fit."""

    os.makedirs(output_dir, exist_ok=True)

    emission_predictions, z_seq = model.predict(emissions, inputs)
    model_ckp = {
        'prefix': model.prefix,
        'copulation_bool': copulation_bool,
        'trained_bool': trained_bool,
        'num_states': model.num_states,
        'learned_params': model.learned_params,
        'learned_lps': model.learned_lps,
        'emissions': emissions,
        'inputs': inputs,
        'state_probs': model.get_state_probs(emissions, inputs),
        'fwd_state_probs': model.get_forward_state_probs(emissions, inputs),
        'emission_predictions': emission_predictions,
        'z_seq': z_seq,
        'score': model.score(emissions, inputs),
        'score_by_o': model.score_by_o(emissions, inputs),
    }

    joblib.dump(model_ckp, os.path.join(output_dir, 'model_ind.pkl'))
    joblib.dump(model.data_config, os.path.join(output_dir, 'data_config.pkl'))
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f: json.dump(model.model_config, f)
    plots.plot_loss(model.learned_lps, savefig=True, fig_dir=output_dir, display=False)
    return


def generate_figures_single(model_dir, savefig=True, display=False, override_fig_dir=True):

    ind_model_ckp, data_config, model_config = load_specific_path_single(model_dir)
    if ind_model_ckp is None: return

    fig_dir = os.path.join(model_dir, 'figures')
    if os.path.exists(fig_dir) and override_fig_dir:
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    learned_params = ind_model_ckp['learned_params']

    plots.plot_var_explained(ind_model_ckp['score'], ind_model_ckp['score'], savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_o(ind_model_ckp['score_by_o'], data_config['emission_labels'],
                                  title='Session', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_prob_states(ind_model_ckp['z_seq'], model_config, title='Session', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_filters(learned_params.emissions.weights, data_config,
                       savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_filter_amplitudes(learned_params.emissions.weights, data_config,
                                 savefig=savefig, fig_dir=fig_dir, display=display)
    return


def generate_figures_all_singles_merged(model_dir, savefig=True, display=False, override_fig_dir=True):
    model_pkls, data_config_pkl, model_config = load_all_singles(model_dir)

    if not model_pkls: return

    fig_dir = os.path.join(model_dir, 'figures')
    if os.path.exists(fig_dir) and override_fig_dir:
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    scores = np.array([mckp['score'] for mckp in model_pkls]) * 200
    print("scores", scores)
    plots.plot_var_explained_ind(scores, title='All sessions', savefig=savefig, fig_dir=fig_dir, display=display)

    fwd_state_probs = [mckp['state_probs'][0] for mckp in model_pkls]
    padded_arrays, n_le = pad_to_equal_length(fwd_state_probs)
    plots.plot_prob_states_aligned(padded_arrays, n_le, 200, model_config, title='All',
                                   xticks=['0', '30'],
                                   xlabel='Time (min)',
                                   savefig=savefig, fig_dir=fig_dir, display=display)

    fwd_state_probs = [mckp['state_probs'][0][:-1] for mckp in model_pkls if mckp['copulation_bool'] is True]
    plots.plot_prob_states_aligned(normalize_to_equal_length(fwd_state_probs, GRID=50000), None, 200, config=model_config, title='All Copulation',
                                   xticks=['Start', 'Copulation'],
                                   xlabel='Time (in courtship)',
                                   savefig=savefig, fig_dir=fig_dir, display=display)

    fwd_state_probs = [mckp['state_probs'][0] for mckp in model_pkls if mckp['copulation_bool'] is False]
    plots.plot_prob_states_aligned(normalize_to_equal_length(fwd_state_probs, GRID=50000), None, 200, config=model_config, title='All No Copulation',
                                   xticks=['0', '30'],
                                   xlabel='Time (min)',
                                   savefig=savefig, fig_dir=fig_dir, display=display)

    return


def save(model, data, train_session_indices, test_session_indices, output_dir):

    os.makedirs(output_dir, exist_ok=False)
    joblib.dump(model.data_config, os.path.join(output_dir, 'data_config.pkl'))
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f: json.dump(model.model_config, f)
    with open(os.path.join(output_dir, 'SUCCESS.txt'), 'w') as f: f.write(str(model.fit_success))

    emissions = data['emissions']
    inputs = data['inputs']
    aux_data = data['aux_data']
    output_mn_std = data['output_mn_std']
    # output_indices = data['output_indices']
    session_keys = np.array(model.data_config['session_keys'])

    train_emissions = [emissions[e] for e in train_session_indices]
    test_emissions = [emissions[e] for e in test_session_indices]
    train_inputs = [inputs[e] for e in train_session_indices]
    test_inputs = [inputs[e] for e in test_session_indices]
    train_aux_data = [aux_data[e] for e in train_session_indices]
    test_aux_data = [aux_data[e] for e in test_session_indices]

    train_lp = model.get_data_logprob(train_emissions, train_inputs)
    test_lp = model.get_data_logprob(test_emissions, test_inputs)

    model_ckp = {
        'prefix': model.prefix,
        'model': model if model.prefix != 'chance' else '',     # chance model cannot unpickle tfd distribution
        'num_states': model.num_states,
        'learned_params': model.learned_params,
        'learned_lps': model.learned_lps,
        'train_data': {
            'train_emissions': train_emissions,
            'train_inputs': train_inputs,
            'train_aux_data': train_aux_data,
            'train_lp': train_lp,
            'train_session_indices': train_session_indices,
            'train_output_mn_std': output_mn_std[train_session_indices],
            'train_session_keys': session_keys[train_session_indices],
            'train_start_frames': data['start_frames'][train_session_indices],
            'train_end_frames': data['end_frames'][train_session_indices],
            'train_downsampled_indices': data['downsampled_indices'][train_session_indices],
            'train_upsampled_indices': data['upsampled_indices'][train_session_indices],
            'train_copulation_bools': np.array(data['copulation_bools'])[train_session_indices]
        },
        'test_data': {
            'test_emissions': test_emissions,
            'test_inputs': test_inputs,
            'test_aux_data': test_aux_data,
            'test_lp': test_lp,
            'test_session_indices': test_session_indices,
            'test_output_mn_std': output_mn_std[test_session_indices],
            'test_session_keys': session_keys[test_session_indices],
            'test_start_frames': data['start_frames'][test_session_indices],
            'test_end_frames': data['end_frames'][test_session_indices],
            'test_downsampled_indices': data['downsampled_indices'][test_session_indices],
            'test_upsampled_indices': data['upsampled_indices'][test_session_indices],
            'test_copulation_bools': np.array(data['copulation_bools'])[test_session_indices]
        },
        # 'output_indices': output_indices,
    }
    joblib.dump(model_ckp, os.path.join(output_dir, 'model_basic.pkl'))
    print("Basic checkpoint dumped.")
    if 'hmm' in model_ckp['prefix']:
        plots.plot_loss(model.learned_lps, savefig=True, fig_dir=output_dir, display=False)
    return


def enhance(output_dir):
    """Load the basic model checkpoint with train and test data and store the full checkpoint enhanced with r2 scores,
    etc. computed."""

    model_ckp = joblib.load(os.path.join(output_dir, 'model_basic.pkl'))
    if model_ckp['prefix'] == 'chance': # Skip predictions etc on the Chance model
        joblib.dump(model_ckp, os.path.join(output_dir, 'model.pkl'))
        return

    model = model_ckp['model']

    def evaluate(prefix_data):
        data_key = f'{prefix_data}_data'

        emissions = model_ckp[data_key][f'{prefix_data}_emissions']
        inputs = model_ckp[data_key][f'{prefix_data}_inputs']
        print(f"Calculating evaluation stats etc on {prefix_data} data...")

        emission_predictions, _ = model.predict_v3(emissions, inputs)
        soft_emission_predictions, z_seq, soft_emission_predictions_per_state = model.predict(emissions, inputs)
        z_probs = model.get_state_probs(emissions, inputs)
        fwd_z_probs = model.get_forward_state_probs(emissions, inputs)

        # from sklearn.metrics import r2_score
        # all_emissions = np.concatenate(emissions, axis=0)
        # all_z_seq = np.concatenate(z_seq, axis=0)
        # all_soft_emission_predictions = np.concatenate(soft_emission_predictions, axis=0)
        # all_emission_predictions = np.concatenate(emission_predictions, axis=0)
        # for z in range(model_ckp['num_states']):
        #     z_mask = all_z_seq == z
        #     print(f"z {z} z_mask", np.sum(z_mask))
        #     r_soft = round(r2_score(all_emissions[z_mask], all_soft_emission_predictions[z_mask]), 3)
        #     r_hard = round(r2_score(all_emissions[z_mask], all_emission_predictions[z_mask]), 3)
        #     for o in range(all_soft_emission_predictions.shape[-1]):
        #         fig = plt.figure( figsize=(20, 6))
        #         ax = plt.gca()
        #         ax.set_title(f'overall r_soft={r_soft} overall r_hard={r_hard}')
        #
        #         ax.plot(all_emissions[z_mask][:10000, o], 'k', label='data')
        #
        #         r = round(r2_score(all_emissions[z_mask][:, o], all_emission_predictions[z_mask][:, o]), 3)
        #         ax.plot(all_emission_predictions[z_mask][:10000, o], 'c-', label=f'hard model (r2={r}) for this o')
        #
        #         r = round(r2_score(all_emissions[z_mask][:, o], all_soft_emission_predictions[z_mask][:, o]), 3)
        #         ax.plot(all_soft_emission_predictions[z_mask][:10000, o], 'm-', label=f'soft model (r2={r}) for this o')
        #         ax.legend(loc='upper right')
        #         plt.suptitle(f'State={z} o={o}')
        #         plt.tight_layout()
        #         plt.show()
        #         plt.close()

        # model_ckp[data_key][f'{prefix_data}_predictions'] = emission_predictions
        model_ckp[data_key][f'{prefix_data}_lp_by_fly'] = model.get_data_logprob_by_fly(emissions, inputs)
        model_ckp[data_key][f'{prefix_data}_soft_predictions'] = soft_emission_predictions
        model_ckp[data_key][f'{prefix_data}_soft_predictions_per_state'] = soft_emission_predictions_per_state
        model_ckp[data_key][f'{prefix_data}_stateseq'] = z_seq
        model_ckp[data_key][f'{prefix_data}_state_probs'] = z_probs
        model_ckp[data_key][f'{prefix_data}_fwd_state_probs'] = fwd_z_probs
        model_ckp[data_key][f'{prefix_data}_score'] = model.score(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix_data}_score_by_fly'] = model.scores_by_fly(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix_data}_score_by_z'] = model.score_by_z(emissions, emission_predictions, z_seq)
        model_ckp[data_key][f'{prefix_data}_score_by_z_soft'] = model.score_by_z_soft(emissions, soft_emission_predictions_per_state, z_probs)
        model_ckp[data_key][f'{prefix_data}_score_by_z_by_fly'] = model.score_by_z_by_fly(emissions, emission_predictions, z_seq)
        model_ckp[data_key][f'{prefix_data}_score_by_z_by_fly_soft'] = model.score_by_z_by_fly_soft(emissions, soft_emission_predictions_per_state, z_probs)

        model_ckp[data_key][f'{prefix_data}_score_by_o'] = model.score_by_o(emissions, emission_predictions)
        model_ckp[data_key][f'{prefix_data}_score_by_o_soft'] = model.score_by_o(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix_data}_score_by_o_by_fly'] = model.score_by_o_by_fly(emissions, emission_predictions)
        model_ckp[data_key][f'{prefix_data}_score_by_o_by_fly_soft'] = model.score_by_o_by_fly(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix_data}_score_by_z_and_o'] = model.score_by_z_and_o(emissions, emission_predictions, z_seq)
        model_ckp[data_key][f'{prefix_data}_score_by_z_and_o_soft'] = model.score_by_z_and_o_soft(emissions, soft_emission_predictions_per_state, z_probs)
        model_ckp[data_key][f'{prefix_data}_score_by_z_and_o_by_fly'] = model.score_by_z_and_o_by_fly(emissions, emission_predictions, z_seq)
        model_ckp[data_key][f'{prefix_data}_score_by_z_and_o_by_fly_soft'] = model.score_by_z_and_o_by_fly_soft(emissions, soft_emission_predictions_per_state, z_probs)
        model_ckp[data_key][f'{prefix_data}_correlation_by_o'] = model.correlation_by_o(emissions, emission_predictions)
        model_ckp[data_key][f'{prefix_data}_correlation_by_o_soft'] = model.correlation_by_o(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix_data}_correlation_by_o_by_fly'] = model.correlation_by_o_by_fly(emissions, emission_predictions)
        model_ckp[data_key][f'{prefix_data}_correlation_by_o_by_fly_soft'] = model.correlation_by_o_by_fly(emissions, soft_emission_predictions)
        return

    evaluate('train')
    evaluate('test')
    joblib.dump(model_ckp, os.path.join(output_dir, 'model.pkl'))
    print("Full checkpoint dumped.")
    return


def generate_figures(model_dir, savefig=True, display=False, override_fig_dir=True):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    fig_dir = os.path.join(model_dir, 'figures')
    if os.path.exists(fig_dir) and override_fig_dir:
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    learned_params = model_ckp['learned_params']
    learned_lps = model_ckp['learned_lps']
    emission_labels = data_config['emission_labels']
    auxiliary_labels = data_config['auxiliary_labels']
    num_states = model_ckp['num_states']
    prefix = model_ckp['prefix']

    if 'hmm' in prefix:
        plots.plot_loss(learned_lps, savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_expected_occupancy(calculate_steady_state_p(learned_params.transitions.transition_matrix),
                                savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_empirical_occupancy(model_ckp['train_data']['train_stateseq'], model_config,
                                title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_ethogram(learned_params.transitions.transition_matrix,
                            savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_transition_matrix(learned_params.transitions.transition_matrix,
                                     savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_ethogram_community(learned_params.transitions.transition_matrix, threshold=0.005,
                                      savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_ethogram_community(learned_params.transitions.transition_matrix, threshold=0.002,
                                      savefig=savefig, fig_dir=fig_dir, display=display)

    weights = learned_params.emissions.weights if 'hmm' in prefix else learned_params['w']
    plots.plot_filters(weights, data_config, savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_filter_amplitudes(weights, data_config, savefig=savefig, fig_dir=fig_dir, display=display)

    plots.plot_var_explained(model_ckp['train_data']['train_score'], model_ckp['test_data']['test_score'], savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_fly(model_ckp['train_data']['train_score_by_fly'], model_ckp['test_data']['test_score_by_fly'], savefig=savefig, fig_dir=fig_dir, display=display)

    plots.plot_ll(model_ckp['train_data']['train_lp'], model_ckp['test_data']['test_lp'], savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_ll_by_fly(model_ckp['train_data']['train_lp_by_fly'], model_ckp['test_data']['test_lp_by_fly'], savefig=savefig, fig_dir=fig_dir, display=display)

    def plot_func(prefix_data):
        data_key = f'{prefix_data}_data'
        emissions = model_ckp[data_key][f'{prefix_data}_emissions']
        aux_data = model_ckp[data_key][f'{prefix_data}_aux_data']
        stateseq = model_ckp[data_key][f'{prefix_data}_stateseq']
        output_mn_std = model_ckp[data_key][f'{prefix_data}_output_mn_std']

        plots.plot_state_mean_outputs_by_o_dists(
            get_emissions_by_state(emissions, stateseq, num_states, rescaled=False),
            emission_labels, title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_state_mean_outputs_by_o_dists(
            get_emissions_by_state(emissions, stateseq, num_states, output_mn_std, rescaled=True),
            emission_labels, title=f'{prefix_data}_ data rescaled (ignore "z-")', savefig=savefig, fig_dir=fig_dir, display=False)
        plots.plot_state_mean_aux_dists(
            get_emissions_by_state(aux_data, stateseq, num_states, rescaled=False),         # reusing get_emissions_by_state func is okay here
            auxiliary_labels, title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z(model_ckp[data_key][f'{prefix_data}_score_by_z'], title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z(model_ckp[data_key][f'{prefix_data}_score_by_z_soft'], title=f'{prefix_data} data (soft)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z_by_fly(model_ckp[data_key][f'{prefix_data}_score_by_z_by_fly'], title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z_by_fly(model_ckp[data_key][f'{prefix_data}_score_by_z_by_fly_soft'], title=f'{prefix_data} data (soft)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_o(model_ckp[data_key][f'{prefix_data}_score_by_o'], emission_labels, title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_o(model_ckp[data_key][f'{prefix_data}_score_by_o_soft'], emission_labels, title=f'{prefix_data} data (soft)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_o_by_fly(model_ckp[data_key][f'{prefix_data}_score_by_o_by_fly'], emission_labels, title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_o_by_fly(model_ckp[data_key][f'{prefix_data}_score_by_o_by_fly_soft'], emission_labels, title=f'{prefix_data} data (soft)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z_o(model_ckp[data_key][f'{prefix_data}_score_by_z_and_o'], emission_labels, title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z_o(model_ckp[data_key][f'{prefix_data}_score_by_z_and_o_soft'], emission_labels, title=f'{prefix_data} data (soft)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z_o_by_fly(model_ckp[data_key][f'{prefix_data}_score_by_z_and_o_by_fly'], emission_labels, title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z_o_by_fly(model_ckp[data_key][f'{prefix_data}_score_by_z_and_o_by_fly_soft'], emission_labels, title=f'{prefix_data} data (soft)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_correlation_by_o(model_ckp[data_key][f'{prefix_data}_correlation_by_o'], emission_labels, title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_correlation_by_o(model_ckp[data_key][f'{prefix_data}_correlation_by_o_soft'], emission_labels, title=f'{prefix_data} data (soft)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_correlation_by_o_by_fly(model_ckp[data_key][f'{prefix_data}_correlation_by_o_by_fly'], emission_labels, title=f'{prefix_data} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_correlation_by_o_by_fly(model_ckp[data_key][f'{prefix_data}_correlation_by_o_by_fly_soft'], emission_labels, title=f'{prefix_data} data (soft)', savefig=savefig, fig_dir=fig_dir, display=display)

        padded_arrays, n_le = pad_to_equal_length(model_ckp[data_key][f'{prefix_data}_state_probs'])
        plots.plot_prob_states_aligned(padded_arrays, n_le, 200, model_config, title=f'All ({prefix_data} data)',
                                       xticks=['0', '30'], xlabel='Time (min)', savefig=savefig, fig_dir=fig_dir, display=display)

        state_probs = [model_ckp[data_key][f'{prefix_data}_state_probs'][i] for i, c in enumerate(model_ckp[data_key][f'{prefix_data}_copulation_bools']) if c == True]
        plots.plot_prob_states_aligned(normalize_to_equal_length(state_probs, GRID=50000), None, 200,
                                       config=model_config, title=f'All Copulation ({prefix_data} data)',
                                       xticks=['Start', 'Copulation'], xlabel='Time (in courtship)', savefig=savefig, fig_dir=fig_dir, display=display)

        padded_arrays, n_le = pad_to_equal_length(model_ckp[data_key][f'{prefix_data}_fwd_state_probs'])
        plots.plot_prob_states_aligned(padded_arrays, n_le, 200, model_config, title=f'All ({prefix_data} data) (fwd)',
                                       xticks=['0', '30'], xlabel='Time (min)', savefig=savefig, fig_dir=fig_dir, display=display)

        state_probs = [model_ckp[data_key][f'{prefix_data}_fwd_state_probs'][i] for i, c in enumerate(model_ckp[data_key][f'{prefix_data}_copulation_bools']) if c == True]
        plots.plot_prob_states_aligned(normalize_to_equal_length(state_probs, GRID=50000), None, 200,
                                       config=model_config, title=f'All Copulation ({prefix_data} data) (fwd)',
                                       xticks=['Start', 'Copulation'], xlabel='Time (in courtship)', savefig=savefig, fig_dir=fig_dir, display=display)
        return

    plot_func('train')
    plot_func('test')
    return


def plot_xlims(model_dir, windows, batch, prefix_data, suffix='', savefig=True, display=False, gen_corr_video=False):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    fig_dir = os.path.join(model_dir, 'figures')
    trajs_dir = os.path.join(fig_dir, 'trajs')
    probs_dir = os.path.join(fig_dir, 'probs')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(trajs_dir, exist_ok=True)
    os.makedirs(probs_dir, exist_ok=True)

    data_key = f'{prefix_data}_data'
    sessions_key = f'{prefix_data}_session_keys'
    dwnsmpl_key = f'{prefix_data}_downsampled_indices'

    effective_fps = data_config['effective_fps']
    key_b = model_ckp[data_key][sessions_key][batch]

    for xlim_ in windows:
        xlim = (int(xlim_[0]), int(xlim_[1]))
        plots.plot_smoothed_probs(model_ckp[data_key][f'{prefix_data}_state_probs'], model_config, batch, effective_fps, xlim=xlim, prefix_data=prefix_data, suffix=suffix, savefig=savefig, fig_path=f'{fig_dir}/probs/{prefix_data}{batch}_xlim={xlim}{suffix}.pdf', display=display)
        plots.plot_comparison_probs(model_ckp[data_key][f'{prefix_data}_state_probs'], model_ckp[data_key][f'{prefix_data}_fwd_state_probs'], model_config, batch, effective_fps, xlim=xlim, prefix_data=prefix_data, suffix=suffix, savefig=savefig, fig_path=f'{fig_dir}/probs/{prefix_data}{batch}_xlim={xlim}{suffix}_.pdf', display=display)
        plots.plot_trajectories(model_ckp, model_config, data_config, batch, prefix_data=prefix_data, suffix=suffix, xlim=xlim, savefig=savefig, fig_path=f'{fig_dir}/trajs/{prefix_data}{batch}_xlim={xlim}{suffix}.pdf', display=display)
        plots.plot_trajectories_w_partner(model_ckp, model_config, data_config, batch, prefix_data=prefix_data, suffix=suffix, xlim=xlim, savefig=savefig, fig_path=f'{fig_dir}/trajs/{prefix_data}{batch}_w_partner_xlim={xlim}{suffix}.pdf', display=display)

        if gen_corr_video:
            xlim_orig = (int(model_ckp[data_key][dwnsmpl_key][batch][xlim[0]]), int(model_ckp[data_key][dwnsmpl_key][batch][xlim[1]]))
            clip_session(os.path.join('/Volumes/murthy/usingla/gold_dataset/wt/mp4', key_b.replace(".h5", ".mp4")), xlim_orig, output_path=f'{fig_dir}/trajs/{prefix_data}{batch}_xlim_orig={xlim_orig}_xlim={xlim}{suffix}.mp4')
    return


def generate_trajs(model_dir, savefig=True, display=False, gen_corr_video=False):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    effective_fps = data_config['effective_fps']

    def f(prefix_data):

        data_key = f'{prefix_data}_data'
        n_sessions = len(model_ckp[data_key][f'{prefix_data}_session_keys'])

        for batch in np.random.choice(n_sessions, size=min(5, n_sessions)):
            batch = 31 if prefix_data == 'train' else batch
            key_b = model_ckp[data_key][f'{prefix_data}_session_keys'][batch]
            num_timestamps = model_ckp[data_key][f'{prefix_data}_stateseq'][batch].shape[0]
            print("batch", batch, "key_b", key_b, "num_timestamps", num_timestamps)
            windows = get_windows_to_plot(effective_fps, num_timestamps)
            # print("windows", windows)
            plot_xlims(model_dir, windows, batch, prefix_data, savefig=savefig, display=display, gen_corr_video=gen_corr_video)
            lastwindows = get_cop_window_to_plot(effective_fps, num_timestamps)
            # print("lastwindows", lastwindows)
            plot_xlims(model_dir, lastwindows, batch, prefix_data, suffix='_nearend', savefig=savefig, display=display, gen_corr_video=gen_corr_video)
            break
    f('train')
    f('test')
    return


# def generate_videos(model_dir, override_vid_dir=True):
#
#     model_ckp, data_config, model_config = load_specific_path(model_dir)
#     if model_ckp is None:
#         return
#
#     vid_dir = os.path.join(model_dir, 'videos')
#     if os.path.exists(vid_dir) and override_vid_dir:
#         shutil.rmtree(vid_dir)
#     os.makedirs(vid_dir, exist_ok=True)
#
#     train_stateseq = model_ckp['train_data']['train_stateseq']
#     train_downsampled_indices = model_ckp['train_data']['train_downsampled_indices']
#     train_upsampled_indices = model_ckp['train_data']['train_upsampled_indices']
#     train_session_keys = model_ckp['train_data']['train_session_keys']
#
#     for batch in np.random.choice(range(len(train_stateseq)), size=min([10, len(train_stateseq)]), replace=False):
#         zseq_b = train_stateseq[batch]
#         downsampled_indices_b = train_downsampled_indices[batch]
#         upsampled_indices_b = train_upsampled_indices[batch]
#         orig_indices_b = downsampled_indices_b[upsampled_indices_b]
#         upsampled_zseq_b = zseq_b[upsampled_indices_b]
#
#         key_b = train_session_keys[batch]
#         intervals_dict_b = get_stateseq_indices(orig_indices_b, upsampled_zseq_b, min_length=150)
#
#         for z in intervals_dict_b:
#             clips_z = intervals_dict_b[z]
#             for interval in random.sample(clips_z, min(10, len(clips_z))):
#                 clip_session(os.path.join('/Volumes/murthy/usingla/gold_dataset/wt/mp4', key_b.replace(".h5", ".mp4")),
#                              interval, output_path=f'{vid_dir}/train{batch}/state{z+1}_origframes={interval}.mp4')
#     return
