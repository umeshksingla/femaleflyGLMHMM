import os
import glob
import joblib
import json
import numpy as np
from functools import partial
from sklearn.metrics import r2_score, mean_squared_error
from wonderwords import RandomWord
from datetime import datetime
from collections import defaultdict

from jax import vmap
from jax.scipy.stats import multivariate_normal
import tensorflow_probability.substrates.jax.distributions as tfd
import jax.numpy as jnp
import jax.random as jnr
import jax

from plotting import plots


def get_data_logprob(hmm, params, emissions, inputs=None):
    """Evaluate the log probability of the data under the given model and model parameters"""
    lp = vmap(partial(hmm.marginal_log_prob, params))(emissions, inputs).sum()
    lp += hmm.log_prior(params)
    lp = lp / emissions.size
    return lp


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


def get_data_logprob_lr(lr, emissions, inputs):
    """
    Linear regression
    P(Y|X, w)
    """
    def fit_normal_residuals(fit_y, true_y):
        residuals = fit_y - true_y
        # print(residuals.shape)
        sigma = jnp.cov(residuals.T)
        mu = jnp.zeros(residuals.shape[-1])
        # print("sigma_tr", sigma)
        # p = multivariate_normal.pdf(residuals, mean=mu, cov=sigma)
        p = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma).prob(residuals)
        # print(p, p.shape)
        p = jnp.maximum(p, 1e-15)
        log_Y_given_wx = jnp.sum(jnp.log(p))
        return log_Y_given_wx

    emissions_pred = jnp.array([lr.predict(_) for _ in inputs])     # session wise
    return vmap(fit_normal_residuals)(emissions_pred, emissions).sum() / emissions.size


def get_data_logprob_mvn(emissions):
    def fit_mvn(y):
        """
        Multivariate gaussian model
        """
        mu = jnp.mean(y, axis=0)
        cov = jnp.cov(y.T)
        # p = multivariate_normal.pdf(y, mean=mu, cov=cov)
        p = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov).prob(y)
        p = jnp.maximum(p, 1e-15)
        log_Y_given_mvn = jnp.sum(jnp.log(p))
        return log_Y_given_mvn

    lp = vmap(fit_mvn)(emissions).sum() / emissions.size
    return lp


def calculate_steady_state_p(P):
    """
    Calculates the steady state probabilities of a Markov chain.

    Parameters:
    P (numpy.ndarray): The transition matrix of the Markov chain.

    Returns:
    numpy.ndarray: The steady state probability vector.
    """
    print(P)
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
    print(steady_state_vector)
    return steady_state_vector


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
    for _ in np.arange(0, length, x_size-x_overlap):
        x_idx_windows.append(np.arange(_, _+x_size))
    x_idx_windows = np.array(x_idx_windows)
    x_idx_windows = x_idx_windows[x_idx_windows[:, -1] < length-y_size-y_gap_size]
    print("x_idx_windows", x_idx_windows, x_idx_windows.shape)

    y_idx_windows = []
    for _ in (x_idx_windows[:, -1] + y_gap_size + 1):
        y_idx_windows.append(np.arange(_, _+y_size))
    y_idx_windows = np.array(y_idx_windows)
    print("y_idx_windows", y_idx_windows, y_idx_windows.shape)
    return x_idx_windows, y_idx_windows


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
    plots.plot_loss(model.learned_lps, savefig=True, fig_dir=output_dir, display=False)

    train_emission_predictions, train_z_predictions = model.predict(train_emissions, train_inputs)
    test_emission_predictions, test_z_predictions = model.predict(test_emissions, test_inputs)

    train_lp = model.get_data_logprob(train_emissions, train_inputs)
    test_lp = model.get_data_logprob(test_emissions, test_inputs)

    model_ckp = {
        'prefix': model.prefix,
        # 'model': model.model,
        'num_states': model.num_states,
        'learned_params': model.learned_params,
        'learned_lps': model.learned_lps,
        'train_data': {
            'train_emissions': train_emissions,
            'train_inputs': train_inputs,
            'train_predictions': train_emission_predictions,
            'train_stateseq': train_z_predictions,
            'train_lp': train_lp,
            'train_score': model.score(train_emissions, train_inputs),
            'train_score_by_o': model.score_by_o(train_emissions, train_inputs),
            'train_score_by_z': model.score_by_z(train_emissions, train_inputs),
            'train_score_by_z_and_o': model.score_by_z_and_o(train_emissions, train_inputs),
            'train_correlation_by_o': model.correlation_by_o(train_emissions, train_inputs),
            'train_session_keys': train_session_keys,
        },
        'test_data': {
            'test_emissions': test_emissions,
            'test_inputs': test_inputs,
            'test_predictions': test_emission_predictions,
            'test_stateseq': test_z_predictions,
            'test_lp': test_lp,
            'test_score': model.score(test_emissions, test_inputs),
            'test_score_by_o': model.score_by_o(test_emissions, test_inputs),
            'test_score_by_z': model.score_by_z(test_emissions, test_inputs),
            'test_score_by_z_and_o': model.score_by_z_and_o(test_emissions, test_inputs),
            'test_correlation_by_o': model.correlation_by_o(test_emissions, test_inputs),
            'test_session_keys': test_session_keys,
        },
        'output_indices': output_indices,
    }
    # print(model_ckp)
    joblib.dump(model_ckp, os.path.join(output_dir, 'model.pkl'))
    return


def generate_figures(model_dir, savefig=True, display=False):

    model_ckp, data_config, model_config = load_specific_path(model_dir)

    fig_dir = os.path.join(model_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    train_stateseq = model_ckp['train_data']['train_stateseq']
    test_stateseq = model_ckp['test_data']['test_stateseq']
    learned_params = model_ckp['learned_params']
    learned_lps = model_ckp['learned_lps']
    emission_labels = data_config['emission_labels']

    plots.plot_loss(learned_lps, savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_prob_states(train_stateseq, model_config, title='train', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_prob_states(test_stateseq, model_config, title='held-out', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(learned_params.transitions.transition_matrix, savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_steady_state(calculate_steady_state_p(learned_params.transitions.transition_matrix), savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_filters(learned_params.emissions.weights, data_config, savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_z(model_ckp['train_data']['train_score_by_z'], title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_z_o(model_ckp['train_data']['train_score_by_z_and_o'], emission_labels, title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_correlation_by_o(model_ckp['train_data']['train_correlation_by_o'], emission_labels, title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_z(model_ckp['test_data']['test_score_by_z'], title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_var_explained_by_z_o(model_ckp['test_data']['test_score_by_z_and_o'], emission_labels, title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_correlation_by_o(model_ckp['test_data']['test_correlation_by_o'], emission_labels, title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)

    os.makedirs(f'{fig_dir}/trajs', exist_ok=True)
    for xlim in [None, (0, 1000), (1500, 2000), (10000, 15000), (0, 5000), (16000, 17000),]:
        for batch in np.random.choice(range(len(train_stateseq)), size=5, replace=False):
            plots.plot_trajectories(model_ckp, model_config, data_config, batch,
                                    prefix_data='train', xlim=xlim, savefig=True,
                                    fig_path=f'{fig_dir}/trajs/train{batch}_xlim={xlim}.pdf',
                                    display=display)

    for xlim in [None, (0, 1000), (1500, 2000), (10000, 15000), (0, 5000), (16000, 17000),]:
        for batch in np.random.choice(range(len(test_stateseq)), size=5, replace=False):
            plots.plot_trajectories(model_ckp, model_config, data_config, batch,
                                    prefix_data='test', xlim=xlim, savefig=True,
                                    fig_path=f'{fig_dir}/trajs/test{batch}_xlim={xlim}.pdf',
                                    display=display)
    print("Done with trajectories.")
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
        raise Warning(f'Unsuccessful model loaded. {model_path}')
    return model_pkl, data_config_pkl, model_config


if __name__ == '__main__':
    create_x_and_y_windows(100, 14, 3, 7, 10)

