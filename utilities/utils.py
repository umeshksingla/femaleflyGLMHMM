import os
import glob
import joblib
import json
import shutil
import numpy as np
from functools import partial
from wonderwords import RandomWord
from datetime import datetime
from collections import defaultdict

# from jax import vmap
# import tensorflow_probability.substrates.jax.distributions as tfd
# import jax.numpy as jnp

from plotting import plots


# def get_data_logprob(hmm, params, emissions, inputs=None):
#     """Evaluate the log probability of the data under the given model and model parameters"""
#     lp = vmap(partial(hmm.marginal_log_prob, params))(emissions, inputs).sum()
#     lp += hmm.log_prior(params)
#     lp = lp / emissions.size
#     return lp


# def get_data_logprob_weightszero(hmm, params, emissions, inputs):
#     """Evaluate the log probability of the data under the given model and model parameters"""
#     print(params.emissions.weights, params.emissions.weights.dtype)
#     emission_params = params.emissions
#     emission_params = params.emissions._replace(
#         weights=jnp.zeros(emission_params.weights.shape),
#         biases=jnp.zeros(emission_params.biases.shape))
#     params = params._replace(emissions=emission_params)
#     lp = vmap(partial(hmm.marginal_log_prob, params))(emissions, inputs).sum()
#     lp += hmm.log_prior(params)
#     lp = lp / emissions.size
#     return lp

# def get_data_logprob_mvn(emissions):
#     def fit_mvn(y):
#         """
#         Multivariate gaussian model
#         """
#         mu = jnp.mean(y, axis=0)
#         cov = jnp.cov(y.T)
#         # p = multivariate_normal.pdf(y, mean=mu, cov=cov)
#         p = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov).prob(y)
#         p = jnp.maximum(p, 1e-15)
#         log_Y_given_mvn = jnp.sum(jnp.log(p))
#         return log_Y_given_mvn

#     lp = vmap(fit_mvn)(emissions).sum() / emissions.size
#     return lp


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


def get_emissions_by_state(emissions, stateseq, num_states):
    emissions_z = {}
    for btch in range(len(stateseq)):
        for z in range(num_states):
            if z not in emissions_z: emissions_z[z] = []
            emissions_z[z].append(emissions[btch][stateseq[btch] == z])
    for z in emissions_z:
        emissions_z[z] = np.vstack(emissions_z[z])
    return emissions_z


def get_stateseq_indices(state_seqs, emissions, min_length=10):
    """

    :param state_seq:
    :param config:
    :param min_length: return indices that are followed by at least min_length frames in the same state
    :return:
    """
    if emissions.ndim <= 2:
        emissions = np.expand_dims(emissions, 0)

    num_timesteps = len(emissions[1])
    num_batches = len(emissions)
    intervals = {}
    for b in range(num_batches):
        intervals[b] = defaultdict(list)
        change_points = np.where(np.diff(state_seqs[b]) != 0)[0]+1
        subseq_starts = np.concatenate(([0], change_points))
        subseq_ends = np.concatenate((change_points, [num_timesteps-1]))
        for s, e in zip(subseq_starts, subseq_ends):
            if (e-s) >= min_length:
                intervals[b][state_seqs[b][s]].append((s, e))
        for z in intervals[b]:
            intervals[b][z] = np.array(intervals[b][z][:-1], dtype=int)
    return intervals


def map_to_video_frame_indices(intervals_dict, output_indices):
    intervals_video_frmindx_dict = {}
    for b in intervals_dict:
        intervals_video_frmindx_dict[b] = {}
        for z in intervals_dict[b]:
            intervals_video_frmindx_dict[b][z] = output_indices[intervals_dict[b][z]]
    return intervals_video_frmindx_dict


def analyze_state_mean(state_seq, config, emissions, inputs):
    """
    Inputs and outputs by state
    :return:
    """
    num_batches = len(emissions)

    # get mean cue in each state
    inputs_z = {}
    outputs_z = {}
    for z in range(config['num_states']):
        inputs_z[z] = []
        outputs_z[z] = []
        for b in range(num_batches):
            z_filter = state_seq[b] == z
            if np.sum(z_filter) == 0:
                print(f'0 samples in this batch for state {z}.')
                continue
            inputs_z[z].append(inputs[b][z_filter])
            outputs_z[z].append(emissions[b][z_filter])
        inputs_z[z] = np.vstack(inputs_z[z])
        outputs_z[z] = np.vstack(outputs_z[z])
    print(inputs_z[0].shape)
    print(outputs_z[0].shape)
    return inputs_z, outputs_z


def get_train_test_split(data, num_fitsessions=None, seed=0, train_frac=0.7):
    print("In get_train_test_split:", num_fitsessions, seed, train_frac)

    data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']
    num_tsessions = data_config['num_sessions']
    if not num_fitsessions:
        num_fitsessions = num_tsessions

    sessions = np.arange(num_tsessions)
    # sessions = jax.random.permutation(jnr.PRNGKey(seed), sessions)[:num_fitsessions]
    print("Permuted sessions", sessions)
    num_train_batches = int(num_fitsessions * train_frac)

    train_session_idxs = sessions[:num_train_batches]
    test_session_idxs = sessions[num_train_batches:]

    train_emissions, train_inputs = emissions[train_session_idxs], inputs[train_session_idxs]
    test_emissions, test_inputs = emissions[test_session_idxs], inputs[test_session_idxs]
    return [train_emissions, train_inputs], [test_emissions, test_inputs]


def save(model, train_emissions, train_inputs, train_session_keys, test_emissions, test_inputs, test_session_keys, output_indices, output_dir):

    os.makedirs(output_dir, exist_ok=False)
    joblib.dump(model.data_config, os.path.join(output_dir, 'data_config.pkl'))
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f: json.dump(model.model_config, f)
    with open(os.path.join(output_dir, 'SUCCESS.txt'), 'w') as f: f.write(str(model.fit_success))

    train_lp = model.get_data_logprob(train_emissions, train_inputs)
    test_lp = model.get_data_logprob(test_emissions, test_inputs)

    model_ckp = {
        'prefix': model.prefix,
        'model': model,
        'num_states': model.num_states,
        'learned_params': model.learned_params,
        'learned_lps': model.learned_lps,
        'train_data': {
            'train_emissions': train_emissions,
            'train_inputs': train_inputs,
            'train_lp': train_lp,
            'train_session_keys': train_session_keys,
        },
        'test_data': {
            'test_emissions': test_emissions,
            'test_inputs': test_inputs,
            'test_lp': test_lp,
            'test_session_keys': test_session_keys,
        },
        'output_indices': output_indices,
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
    model = model_ckp['model']

    train_emissions = model_ckp['train_data']['train_emissions']
    train_inputs = model_ckp['train_data']['train_inputs']
    test_emissions = model_ckp['test_data']['test_emissions']
    test_inputs = model_ckp['test_data']['test_inputs']

    #### calculate evaluation stats etc on train and test data
    train_emission_predictions, train_z_seq = model.predict(train_emissions, train_inputs)
    test_emission_predictions, test_z_seq = model.predict(test_emissions, test_inputs)
    train_z_probs = model.get_state_probs(train_emissions, train_inputs)
    test_z_probs = model.get_state_probs(test_emissions, test_inputs)
    train_fwd_z_probs = model.get_forward_state_probs(train_emissions, train_inputs)
    test_fwd_z_probs = model.get_forward_state_probs(test_emissions, test_inputs)

    model_ckp['train_data']['train_predictions'] = train_emission_predictions
    model_ckp['train_data']['train_stateseq'] = train_z_seq
    model_ckp['train_data']['train_state_probs'] = train_z_probs
    model_ckp['train_data']['train_fwd_state_probs'] = train_fwd_z_probs
    model_ckp['train_data']['train_score'] = model.score(train_emissions, train_inputs)
    model_ckp['train_data']['train_score_by_o'] = model.score_by_o(train_emissions, train_inputs)
    model_ckp['train_data']['train_score_by_z'] = model.score_by_z(train_emissions, train_inputs)
    model_ckp['train_data']['train_score_by_z_and_o'] = model.score_by_z_and_o(train_emissions, train_inputs)
    model_ckp['train_data']['train_correlation_by_o'] = model.correlation_by_o(train_emissions, train_inputs)

    model_ckp['test_data']['test_predictions'] = test_emission_predictions
    model_ckp['test_data']['test_stateseq'] = test_z_seq
    model_ckp['test_data']['test_state_probs'] = test_z_probs
    model_ckp['test_data']['test_fwd_state_probs'] = test_fwd_z_probs
    model_ckp['test_data']['test_score'] = model.score(test_emissions, test_inputs)
    model_ckp['test_data']['test_score_by_o'] = model.score_by_o(test_emissions, test_inputs)
    model_ckp['test_data']['test_score_by_z'] = model.score_by_z(test_emissions, test_inputs)
    model_ckp['test_data']['test_score_by_z_and_o'] = model.score_by_z_and_o(test_emissions, test_inputs)
    model_ckp['test_data']['test_correlation_by_o'] = model.correlation_by_o(test_emissions, test_inputs)

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

    train_stateseq = model_ckp['train_data']['train_stateseq']
    test_stateseq = model_ckp['test_data']['test_stateseq']
    learned_params = model_ckp['learned_params']
    learned_lps = model_ckp['learned_lps']
    emission_labels = data_config['emission_labels']
    num_states = model_ckp['num_states']


    plots.plot_expected_occupancy(calculate_steady_state_p(learned_params.transitions.transition_matrix),
                            savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_empirical_occupancy(train_stateseq, model_config,
                            title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
    return

    plots.plot_ethogram(learned_params.transitions.transition_matrix,
                        savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(learned_params.transitions.transition_matrix,
                                 savefig=savefig, fig_dir=fig_dir, display=display)
    # returns

    plots.plot_filters(learned_params.emissions.weights, data_config,
                       savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_filter_amplitudes(learned_params.emissions.weights, data_config,
                       savefig=savefig, fig_dir=fig_dir, display=display)
    # return

    # btch=32
    # plots.plot_smoothed_probs(model_ckp['train_data']['train_state_probs'], model_config, btch, effective_fps=data_config["predict_window_size"],
    #                           xlim=(0, 19907),
    #                           prefix_data='train', savefig=savefig,
    #                           fig_path=f'{fig_dir}/probs/train{btch}.pdf', display=display)
    # plots.plot_trajectories(model_ckp, model_config, data_config, btch,
    #                         prefix_data='train', xlim=(0, 19907), savefig=savefig,
    #                         fig_path=f'{fig_dir}/trajs/train{btch}.pdf',
    #                         display=display)

    plots.plot_var_explained(model_ckp['train_data']['train_score'], model_ckp['test_data']['test_score'],
                             savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_ethogram_community(learned_params.transitions.transition_matrix, threshold=0.005,
                                  savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_ethogram_community(learned_params.transitions.transition_matrix, threshold=0.002,
                                  savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_loss(learned_lps, savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_prob_states(train_stateseq, model_config,
                           title='train', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_prob_states(test_stateseq, model_config,
                           title='held-out', savefig=savefig, fig_dir=fig_dir, display=display)

    plots.plot_state_mean_outputs_by_o_dists(get_emissions_by_state(model_ckp['train_data']['train_emissions'], train_stateseq, num_states), emission_labels,
                                             title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_z(model_ckp['train_data']['train_score_by_z'],
                                  title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_z_o(model_ckp['train_data']['train_score_by_z_and_o'], emission_labels,
                                    title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_correlation_by_o(model_ckp['train_data']['train_correlation_by_o'], emission_labels,
                                title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)

    plots.plot_state_mean_outputs_by_o_dists(get_emissions_by_state(model_ckp['test_data']['test_emissions'], test_stateseq, num_states), emission_labels,
                                             title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_z(model_ckp['test_data']['test_score_by_z'],
                                  title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_z_o(model_ckp['test_data']['test_score_by_z_and_o'], emission_labels,
                                    title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_correlation_by_o(model_ckp['test_data']['test_correlation_by_o'], emission_labels,
                                title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)

    os.makedirs(f'{fig_dir}/trajs', exist_ok=True)
    os.makedirs(f'{fig_dir}/probs', exist_ok=True)

    effective_fps = data_config["predict_window_size"]
    window_size = effective_fps * 1 * 60    # 1 minute windows  # TODO TODO FIX!!!!
    num_timestamps = train_stateseq.shape[1]

    window_starts = np.linspace(0, num_timestamps-window_size-10, num=10)
    window_starts = np.round(window_starts).astype(int)
    window_ends = window_starts + window_size     # 1 min windows

    window_starts2 = np.linspace(0, num_timestamps-window_size*5-10, num=5)
    window_starts2 = np.round(window_starts2).astype(int)
    window_ends2 = window_starts2 + window_size*5   # 5 min windows

    window_starts3 = np.linspace(0, num_timestamps-window_size//60-10, num=5)
    window_starts3 = np.round(window_starts3).astype(int)
    window_ends3 = window_starts3 + window_size//60   # 1 sec windows

    all_window_starts = np.hstack((window_starts, window_starts2, window_starts3))
    all_window_ends = np.hstack((window_ends, window_ends2, window_ends3))

    all_window_starts = np.append(all_window_starts, 0)     # add full session window
    all_window_ends = np.append(all_window_ends, num_timestamps-1)

    for batch in np.random.choice(range(len(train_stateseq)), size=min([5, len(train_stateseq)]), replace=False):
        for xlim in zip(all_window_starts, all_window_ends):  # on train
            plots.plot_smoothed_probs(model_ckp['train_data']['train_state_probs'], model_config, batch, effective_fps, xlim=xlim,
                                      prefix_data='train', savefig=savefig, fig_path=f'{fig_dir}/probs/train{batch}_xlim={xlim}.pdf', display=display)
            plots.plot_comparison_probs(model_ckp['train_data']['train_state_probs'], model_ckp['train_data']['train_fwd_state_probs'], model_config, batch, effective_fps,
                                      xlim=xlim,
                                      prefix_data='train', savefig=savefig,
                                      fig_path=f'{fig_dir}/probs/train{batch}_xlim={xlim}.pdf', display=display)
            plots.plot_trajectories(model_ckp, model_config, data_config, batch,
                                    prefix_data='train', xlim=xlim, savefig=savefig,
                                    fig_path=f'{fig_dir}/trajs/train{batch}_xlim={xlim}_.pdf',
                                    display=display)
            break
        break

    # for batch in np.random.choice(range(len(test_stateseq)), size=min([5, len(test_stateseq)]), replace=False):
    for batch in range(len(test_stateseq)):
        for xlim in zip(all_window_starts, all_window_ends):  # on test
            plots.plot_smoothed_probs(model_ckp['test_data']['test_state_probs'], model_config, batch, effective_fps,
                                      xlim=xlim,
                                      prefix_data='test', savefig=savefig,
                                      fig_path=f'{fig_dir}/probs/test{batch}_xlim={xlim}.pdf', display=display)
            plots.plot_comparison_probs(model_ckp['test_data']['test_state_probs'], model_ckp['test_data']['test_fwd_state_probs'], model_config, batch, effective_fps,
                                      xlim=xlim,
                                      prefix_data='test', savefig=savefig,
                                      fig_path=f'{fig_dir}/probs/test{batch}_xlim={xlim}.pdf', display=display)
            plots.plot_trajectories(model_ckp, model_config, data_config, batch,
                                    prefix_data='test', xlim=xlim, savefig=savefig,
                                    fig_path=f'{fig_dir}/trajs/test{batch}_xlim={xlim}_.pdf',
                                    display=display)
            break
        break
    return


def generate_figures2(model_dir, data_pkl_path, savefig=True, display=False):
    """Figures that need raw data to be loaded, such as male data to define contexts"""

    model_ckp, _, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    data = joblib.load(data_pkl_path)
    data_config, emissions, inputs, aux_data = data['data_config'], data['emissions'], data['inputs'], data['aux_data']
    # todo save aux data with model as well
    num_batches = data_config['num_sessions']
    num_train_batches = int(num_batches * 0.8)
    train_emissions, train_inputs, train_auxs = emissions[:num_train_batches], inputs[:num_train_batches], aux_data[:num_train_batches]
    # test_emissions, test_inputs, test_auxs = emissions[num_train_batches:], inputs[num_train_batches:], aux_data[num_train_batches:]
    print("num_batches", num_batches, "num_train_batches", num_train_batches)
    print(train_emissions.shape, train_inputs.shape, train_auxs.shape)

    train_stateseq = model_ckp['train_data']['train_stateseq']
    # test_stateseq = model_ckp['test_data']['test_stateseq']
    learned_params = model_ckp['learned_params']
    # learned_lps = model_ckp['learned_lps']
    # emission_labels = data_config['emission_labels']
    num_states = model_ckp['num_states']

    fig_dir = os.path.join(model_dir, 'figures')
    plots.plot_state_mean_aux(train_auxs, train_stateseq, num_states, data_config,
                                 title='Train', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_mean_outs(train_emissions, train_stateseq, num_states, data_config,
                                 title='Train', savefig=savefig, fig_dir=fig_dir, display=display)
    return


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

