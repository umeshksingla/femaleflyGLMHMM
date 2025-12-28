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
from itertools import groupby

from glm_utils.preprocessing import BasisProjection
import tensorflow_probability.substrates.jax.distributions as tfd
import jax.numpy as jnp


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


def load_specific_path_auxem(model_path):
    print('Loading auxem:', model_path)
    model_pkl = joblib.load(os.path.join(model_path, 'auxem_model.pkl'))
    with open(os.path.join(model_path, 'SUCCESS.txt')) as f: fit_success = f.read()
    if fit_success != 'True':
        print(Warning(f'Unsuccessful model loaded. {model_path}'))
        return None
    return model_pkl


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


def get_emissions_by_state(emissions, stateseq, num_states, output_mn_std=None, rescaled=False, effective_fps=None):
    """
    Return a dictionary of states mapped to emission values in that state.
    :param emissions:
    :param stateseq:
    :param num_states:
    :param output_mn_std:
    :param rescaled:
    :param effective_fps:
    :return:
    """
    emissions_z = {}
    for btch in range(len(stateseq)):
        for z in range(num_states):
            if z not in emissions_z: emissions_z[z] = []
            if rescaled:
                mn_std_btch = output_mn_std[btch]
                eez = emissions[btch][stateseq[btch] == z] * mn_std_btch[:, 1, None].T + mn_std_btch[:, 0, None].T
                eez = eez * effective_fps  # velocity (mm/s) = velocity (mm/frame) × effective_fps
            else:
                eez = emissions[btch][stateseq[btch] == z]
            emissions_z[z].append(eez)
    for z in emissions_z:
        emissions_z[z] = np.vstack(emissions_z[z])
    return emissions_z


def get_rescaled_emissions(emissions, output_mn_std, effective_fps):
    emissions_rescaled_b = []
    for btch in range(len(emissions)):
        mn_std_btch = output_mn_std[btch]
        eez = emissions[btch] * mn_std_btch[:, 1, None].T + mn_std_btch[:, 0, None].T
        eez = eez * effective_fps  # velocity (mm/s) = velocity (mm/frame) × effective_fps
        emissions_rescaled_b.append(eez)
    return emissions_rescaled_b


def get_aux_by_state(aux_data, stateseq, num_states, aux_mn_std=None, rescaled=False, effective_fps=None):
    """
    Return a dictionary of states mapped to emission values in that state.
    :param aux_data:
    :param stateseq:
    :param num_states:
    :param output_mn_std:
    :param rescaled:
    :param effective_fps:
    :return:
    """
    aux_data_z = {}
    for btch in range(len(stateseq)):
        for z in range(num_states):
            if z not in aux_data_z: aux_data_z[z] = []
            if rescaled:
                mn_std_btch = aux_mn_std[btch]
                eez = aux_data[btch][stateseq[btch] == z] * mn_std_btch[:, 1, None].T + mn_std_btch[:, 0, None].T
                eez = eez #* effective_fps
            else:
                eez = aux_data[btch][stateseq[btch] == z]
            aux_data_z[z].append(eez)
    for z in aux_data_z:
        aux_data_z[z] = np.vstack(aux_data_z[z])
    return aux_data_z


def get_rescaled_aux(aux_data, aux_mn_std):
    aux_data_rescaled_b = []
    for btch in range(len(aux_data)):
        mn_std_btch = aux_mn_std[btch]
        eez = aux_data[btch] * mn_std_btch[:, 1, None].T + mn_std_btch[:, 0, None].T
        aux_data_rescaled_b.append(eez)
    return aux_data_rescaled_b


def get_feat_windows(true_feat_series, pred_feat_series, event_onsets, window_len):
    """

    :param true_feat_series:
    :param pred_feat_series:
    :param event_onsets:
    :param window_len:
    :return:
    """
    true_feat_series_windows = []
    pred_feat_series_windows = []

    for onset in event_onsets:
        start = onset - window_len
        end = onset + window_len
        if (start < 0) or (end > len(true_feat_series)):
            continue  # skip this window

        t_win = true_feat_series[start:end]
        p_win = pred_feat_series[start:end]

        true_feat_series_windows.append(t_win)
        pred_feat_series_windows.append(p_win)
    true_feat_series_windows = np.stack(true_feat_series_windows)  # shape (n_onsets, 2 * window_frames)
    pred_feat_series_windows = np.stack(pred_feat_series_windows)  # shape (n_onsets, 2 * window_frames)
    return true_feat_series_windows, pred_feat_series_windows


def get_event_onsets(eventts, min_duration, lr_mask=None):
    # print(np.unique(eventts, return_counts=True))
    eventts = np.where(eventts > 0, 1, 0)   # replace pulse, sine, tap in original binary space (they are zscored)
    if lr_mask is not None:
        eventts = eventts[lr_mask]
    event_onsets = np.where((eventts[1:] == 1) & (eventts[:-1] == 0))[0] + 1
    event_offsets = np.where((eventts[1:] == 0) & (eventts[:-1] == 1))[0] + 1
    if eventts[0] == 1:
        event_onsets = np.insert(event_onsets, 0, 0)
    if eventts[-1] == 1:
        event_offsets = np.append(event_offsets, len(eventts) - 1)
    durations = event_offsets - event_onsets  # in frames
    keep_idx = np.where(durations >= min_duration)[0]
    filtered_onsets = event_onsets[keep_idx]
    return filtered_onsets


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
        max_start = max(0, num_timestamps - window_duration_frames - 1)    # to avoid linspace going negative
        if max_start == 0:
            return np.empty((0, 2), dtype=int)
        starts = np.linspace(0, max_start, num=n_windows)
        starts = np.round(starts).astype(int)
        ends = starts + window_duration_frames
        windows = np.stack((starts, ends), axis=1)
        return windows

    window_size = effective_fps * 1  # 1 second windows
    windows1 = make_windows(window_size, n_windows=20)  # 1 sec
    windows2 = make_windows(window_size * 30, n_windows=10)  # 30 sec
    windows3 = make_windows(window_size * 60, n_windows=10)  # 1 min
    windows4 = make_windows(window_size * 60 * 5, n_windows=5)  # 5 min

    windows = np.vstack((windows1, windows2, windows3, windows4)).astype(int)

    return windows


def get_cop_window_to_plot(effective_fps, num_timestamps):

    window_size = effective_fps * 1  # 1 second windows
    last_window_size = window_size * 30   # Last 30-second window, to visualize near-copulation
    if num_timestamps > last_window_size:
        last_window = np.array([[num_timestamps - last_window_size, num_timestamps - 1]])
    else:
        last_window = np.empty((0, 2), dtype=int)
    windows = np.vstack((last_window)).astype(int)
    return windows


def get_full_window_to_plot(effective_fps, num_timestamps):
    full_session_window = np.array([[0, num_timestamps - 1]])  # Full session window
    half_session_window1 = np.array([[0, num_timestamps//2 - 1]])  # Full session window
    half_session_window2 = np.array([[num_timestamps//2, num_timestamps - 1]])  # Full session window
    windows = np.vstack((full_session_window, half_session_window1, half_session_window2)).astype(int)
    return windows


def get_chance_logprob(y):
    mu = jnp.mean(y, axis=0)
    cov = jnp.cov(y.T)
    cov = jnp.atleast_2d(cov)
    print(mu, cov)
    model = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
    p = model.prob(y)
    p = jnp.maximum(p, 1e-15)
    log_Y_given_mvn = jnp.sum(jnp.log(p))
    lp = log_Y_given_mvn.sum()
    return lp


def calc_dwell_times_by_z(z_seqs, num_states):
    print("z_seqs", len(z_seqs))
    dwell_times_z = {z: [] for z in range(num_states)}
    for z_seq in z_seqs:
        for z, group in groupby(np.array(z_seq)):
            dwell_times_z[z].append(len(list(group)))

    for z in dwell_times_z:
        dwell_times_z[z] = np.array(dwell_times_z[z])
    return dwell_times_z


def get_state_indices(z_seq, z, min_length=5, max_length=None, max_clips=5):
    state_sequence = np.array(z_seq)
    is_target = (state_sequence == z).astype(int)
    diff = np.diff(is_target, prepend=0, append=0)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    lengths = ends - starts + 1
    keep = (lengths >= min_length)
    if max_length:
        keep = (lengths >= min_length) & (lengths <= max_length)
    clips = np.stack([starts[keep], ends[keep]], axis=1)

    if len(clips) > max_clips:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(clips), size=max_clips, replace=False)
        clips = clips[indices]

    return clips


def basis_invtransform_one_by_one(weights, basis, n_inputs):
    num_states = weights.shape[0]
    emission_dim = weights.shape[1]
    filter_len = weights.shape[2]
    each_filter_len = filter_len // n_inputs
    new_weights = []
    for z in range(num_states):
        z_ = []
        for o in range(emission_dim):
            o_ = []
            for e in range(n_inputs):
                w_ = weights[z][o, e * each_filter_len:e * each_filter_len + each_filter_len]
                # print(w_, w_.shape)
                w = BasisProjection(basis).inverse_transform(w_).squeeze()
                # print(w.shape)
                o_.append(w)
            # print(o_)
            o_ = np.array(o_).reshape(-1)
            # print(o_)
            z_.append(o_)
        new_weights.append(z_)
    new_weights = np.array(new_weights)
    return new_weights.reshape(num_states, emission_dim, n_inputs, -1)
