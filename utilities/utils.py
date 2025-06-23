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
from utilities.io import *


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
    # model_ckp_enhanced = enhance(model_ckp=model_ckp)
    # del model.model     # delete inner dynamax etc classes
    joblib.dump(model_ckp, os.path.join(output_dir, 'model_basic.pkl'))
    # joblib.dump(model_ckp_enhanced, os.path.join(output_dir, 'model_enhanced.pkl'))
    # full_model = dict()
    # full_model.update(model_ckp)
    # full_model.update(model_ckp_enhanced)
    # joblib.dump(full_model, os.path.join(output_dir, 'model.pkl'))
    print("Basic checkpoint dumped.")
    if 'hmm' in model_ckp['prefix']:
        plots.plot_loss(model.learned_lps, savefig=True, fig_dir=output_dir, display=False)
    return


def enhance(output_dir=None, model_ckp=None):
    """Load or use the basic model checkpoint with train and test data and store the full checkpoint enhanced with r2 scores,
    etc. computed."""

    if not model_ckp:
        model_ckp = joblib.load(os.path.join(output_dir, 'model_basic.pkl'))

    if model_ckp['prefix'] == 'chance': # Skip predictions etc on the Chance model
        joblib.dump(model_ckp, os.path.join(output_dir, 'model.pkl'))
        return

    model = model_ckp['model']

    def evaluate(prefix_data):
        data_key = f'{prefix_data}_data'

        emissions = model_ckp[data_key][f'{prefix_data}_emissions']
        inputs = model_ckp[data_key][f'{prefix_data}_inputs']
        print(f"Calculating enhanced stats etc on {prefix_data} data...")

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
    if output_dir:
        joblib.dump(model_ckp, os.path.join(output_dir, 'model.pkl'))
        print("Full checkpoint dumped.")
    return model_ckp


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
    weights = weights[:, np.newaxis, :] if 'logrhmm' in prefix else weights
    print("weights shape", learned_params.emissions.weights.shape, learned_params.emissions.biases.shape)
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
