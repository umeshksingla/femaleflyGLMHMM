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