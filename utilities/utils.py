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
from collections import defaultdict, OrderedDict

from plotting import plots
from utilities.video_utils import clip_session
from utilities.io import *
from utilities.logreg import train_aux_emissions, predict_aux_emissions


# def save_single(model, emissions, inputs, copulation_bool, trained_bool, output_dir):
#     """Save single session fit."""
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     emission_predictions, z_seq = model.predict(emissions, inputs)
#     model_ckp = {
#         'prefix': model.prefix,
#         'copulation_bool': copulation_bool,
#         'trained_bool': trained_bool,
#         'num_states': model.num_states,
#         'learned_params': model.learned_params,
#         'learned_lps': model.learned_lps,
#         'emissions': emissions,
#         'inputs': inputs,
#         'state_probs': model.get_state_probs(emissions, inputs),
#         'fwd_state_probs': model.get_forward_state_probs(emissions, inputs),
#         'emission_predictions': emission_predictions,
#         'z_seq': z_seq,
#         'score': model.score(emissions, inputs),
#         'score_by_o': model.score_by_o(emissions, inputs),
#     }
#
#     joblib.dump(model_ckp, os.path.join(output_dir, 'model_ind.pkl'))
#     joblib.dump(model.data_config, os.path.join(output_dir, 'data_config.pkl'))
#     with open(os.path.join(output_dir, 'model_config.json'), 'w') as f: json.dump(model.model_config, f)
#     plots.plot_loss(model.learned_lps, savefig=True, fig_dir=output_dir, display=False)
#     return


# def generate_figures_single(model_dir, savefig=True, display=False, override_fig_dir=True):
#
#     ind_model_ckp, data_config, model_config = load_specific_path_single(model_dir)
#     if ind_model_ckp is None: return
#
#     fig_dir = os.path.join(model_dir, 'figures')
#     if os.path.exists(fig_dir) and override_fig_dir:
#         shutil.rmtree(fig_dir)
#     os.makedirs(fig_dir, exist_ok=True)
#
#     learned_params = ind_model_ckp['learned_params']
#
#     plots.plot_var_explained(ind_model_ckp['score'], ind_model_ckp['score'], savefig=savefig, fig_dir=fig_dir, display=display)
#     plots.plot_var_explained_by_o(ind_model_ckp['score_by_o'], data_config['emission_labels'],
#                                   title='Session', savefig=savefig, fig_dir=fig_dir, display=display)
#     plots.plot_prob_states(ind_model_ckp['z_seq'], model_config, title='Session', savefig=savefig, fig_dir=fig_dir, display=display)
#     plots.plot_filters(learned_params.emissions.weights, data_config,
#                        savefig=savefig, fig_dir=fig_dir, display=display)
#     plots.plot_filter_amplitudes(learned_params.emissions.weights, data_config,
#                                  savefig=savefig, fig_dir=fig_dir, display=display)
#     return


# def generate_figures_all_singles_merged(model_dir, savefig=True, display=False, override_fig_dir=True):
#     model_pkls, data_config_pkl, model_config = load_all_singles(model_dir)
#
#     if not model_pkls: return
#
#     fig_dir = os.path.join(model_dir, 'figures')
#     if os.path.exists(fig_dir) and override_fig_dir:
#         shutil.rmtree(fig_dir)
#     os.makedirs(fig_dir, exist_ok=True)
#
#     scores = np.array([mckp['score'] for mckp in model_pkls]) * 200
#     print("scores", scores)
#     plots.plot_var_explained_ind(scores, title='All sessions', savefig=savefig, fig_dir=fig_dir, display=display)
#
#     fwd_state_probs = [mckp['state_probs'][0] for mckp in model_pkls]
#     padded_arrays, n_le = pad_to_equal_length(fwd_state_probs)
#     plots.plot_prob_states_aligned(padded_arrays, n_le, 200, model_config, title='All',
#                                    xticks=['0', '30'],
#                                    xlabel='Time (min)',
#                                    savefig=savefig, fig_dir=fig_dir, display=display)
#
#     fwd_state_probs = [mckp['state_probs'][0][:-1] for mckp in model_pkls if mckp['copulation_bool'] is True]
#     plots.plot_prob_states_aligned(normalize_to_equal_length(fwd_state_probs, GRID=50000), None, 200, config=model_config, title='All Copulation',
#                                    xticks=['Start', 'Copulation'],
#                                    xlabel='Time (in courtship)',
#                                    savefig=savefig, fig_dir=fig_dir, display=display)
#
#     fwd_state_probs = [mckp['state_probs'][0] for mckp in model_pkls if mckp['copulation_bool'] is False]
#     plots.plot_prob_states_aligned(normalize_to_equal_length(fwd_state_probs, GRID=50000), None, 200, config=model_config, title='All No Copulation',
#                                    xticks=['0', '30'],
#                                    xlabel='Time (min)',
#                                    savefig=savefig, fig_dir=fig_dir, display=display)
#
#     return


def save(model, data, train_session_indices, test_session_indices, output_dir):

    os.makedirs(output_dir, exist_ok=False)
    joblib.dump(model.data_config, os.path.join(output_dir, 'data_config.pkl'))
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f: json.dump(model.model_config, f)
    with open(os.path.join(output_dir, 'SUCCESS.txt'), 'w') as f: f.write(str(model.fit_success))

    emissions = data['emissions']
    inputs = data['inputs']
    aux_data = data['aux_data']
    aux_emissions = data['aux_emissions']
    output_mn_std = data['output_mn_std']
    aux_mn_std = data['aux_mn_std']
    session_keys = np.array(model.data_config['session_keys'])

    train_emissions = [emissions[e] for e in train_session_indices]
    test_emissions = [emissions[e] for e in test_session_indices]
    train_inputs = [inputs[e] for e in train_session_indices]
    test_inputs = [inputs[e] for e in test_session_indices]
    train_aux_data = [aux_data[e] for e in train_session_indices]
    test_aux_data = [aux_data[e] for e in test_session_indices]
    train_aux_emissions = [aux_emissions[e] for e in train_session_indices]
    test_aux_emissions = [aux_emissions[e] for e in test_session_indices]

    train_lp = model.get_data_logprob(train_emissions, train_inputs)
    train_lps_by_fly = model.get_data_logprob_by_fly(train_emissions, train_inputs)
    test_lp = model.get_data_logprob(test_emissions, test_inputs)
    test_lps_by_fly = model.get_data_logprob_by_fly(test_emissions, test_inputs)

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
            'train_aux_emissions': train_aux_emissions,
            'train_lp': train_lp,
            'train_lps_by_fly': train_lps_by_fly,
            'train_session_indices': train_session_indices,
            'train_output_mn_std': output_mn_std[train_session_indices],
            'train_aux_mn_std': aux_mn_std[train_session_indices],
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
            'test_aux_emissions': test_aux_emissions,
            'test_lp': test_lp,
            'test_lps_by_fly': test_lps_by_fly,
            'test_session_indices': test_session_indices,
            'test_output_mn_std': output_mn_std[test_session_indices],
            'test_aux_mn_std': aux_mn_std[test_session_indices],
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


def enhance(output_dir=None, model_ckp=None):
    """Load or use the basic model checkpoint with train and test data and store the full checkpoint enhanced with r2 scores,
    etc. computed."""

    if not model_ckp:
        model_ckp = joblib.load(os.path.join(output_dir, 'model_basic.pkl'))

    if model_ckp['prefix'] == 'chance': # Skip predictions etc on the Chance model
        joblib.dump(model_ckp, os.path.join(output_dir, 'model.pkl'))
        return

    model = model_ckp['model']

    def evaluate(prefix):
        data_key = f'{prefix}_data'

        emissions = model_ckp[data_key][f'{prefix}_emissions']
        aux_emissions = model_ckp[data_key][f'{prefix}_aux_emissions']
        inputs = model_ckp[data_key][f'{prefix}_inputs']
        print(f"Calculating enhanced stats etc on {prefix} data...")

        # emission_predictions, _ = model.predict_v3(emissions, inputs)
        soft_emission_predictions, z_seqs, soft_emission_predictions_per_state = model.predict(emissions, inputs)
        z_probs = model.get_state_probs(emissions, inputs)
        fwd_z_probs = model.get_forward_state_probs(emissions, inputs)

        if prefix == 'train':
            # accuracy, counts, w, b = train_aux_emissions(inputs, aux_emissions, z_seqs, model.num_states)
            w, b, probs = train_aux_emissions(inputs, aux_emissions, z_seqs, model.num_states)
            # model_ckp[data_key][f'{prefix}_auxem_acc_scores_z_and_o'] = accuracy
            # model_ckp[data_key][f'{prefix}_auxem_eventcounts_z_and_o'] = counts
            model_ckp['logreg_params'] = {'w': w, 'b': b}
        elif prefix == 'test':
            # accuracy, counts = predict_aux_emissions(model_ckp['logreg_params'], inputs, aux_emissions, z_seqs, model.num_states)
            probs = predict_aux_emissions(model_ckp['logreg_params'], inputs, aux_emissions, z_seqs, model.num_states)
        #     model_ckp[data_key][f'{prefix}_auxem_acc_scores_z_and_o'] = accuracy
        #     model_ckp[data_key][f'{prefix}_auxem_eventcounts_z_and_o'] = counts
        else:
            raise Exception('huh')
        model_ckp[data_key][f'{prefix}_probs_z_and_auxo'] = probs

        # model_ckp[data_key][f'{prefix}_predictions'] = emission_predictions
        # model_ckp[data_key][f'{prefix}_lp_again'] = model.get_data_logprob(emissions, inputs)
        # model_ckp[data_key][f'{prefix}_lp_by_fly_again'] = model.get_data_logprob_by_fly(emissions, inputs)
        model_ckp[data_key][f'{prefix}_soft_predictions'] = soft_emission_predictions
        model_ckp[data_key][f'{prefix}_soft_predictions_per_state'] = soft_emission_predictions_per_state
        model_ckp[data_key][f'{prefix}_stateseq'] = z_seqs
        model_ckp[data_key][f'{prefix}_state_probs'] = z_probs
        model_ckp[data_key][f'{prefix}_fwd_state_probs'] = fwd_z_probs

        model_ckp[data_key][f'{prefix}_score'] = model.score(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix}_score_by_fly'] = model.scores_by_fly(emissions, soft_emission_predictions)
        # model_ckp[data_key][f'{prefix}_score_by_z_soft'] = model.score_by_z_soft(emissions, soft_emission_predictions_per_state, z_probs)
        model_ckp[data_key][f'{prefix}_score_by_z_by_fly_soft'] = model.score_by_z_by_fly_soft(emissions, soft_emission_predictions_per_state, z_probs)
        # model_ckp[data_key][f'{prefix}_score_by_o_soft'] = model.score_by_o(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix}_score_by_o_by_fly_soft'] = model.score_by_o_by_fly(emissions, soft_emission_predictions)
        # model_ckp[data_key][f'{prefix}_score_by_z_and_o_soft'] = model.score_by_z_and_o_soft(emissions, soft_emission_predictions_per_state, z_probs)
        model_ckp[data_key][f'{prefix}_score_by_z_and_o_by_fly_soft'] = model.score_by_z_and_o_by_fly_soft(emissions, soft_emission_predictions_per_state, z_probs)

        model_ckp[data_key][f'{prefix}_pearson'] = model.pearson(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix}_pearson_by_fly'] = model.pearson_by_fly(emissions, soft_emission_predictions)
        # model_ckp[data_key][f'{prefix}_pearson_by_z'] = model.pearson_by_z(emissions, soft_emission_predictions_per_state, z_probs)
        model_ckp[data_key][f'{prefix}_pearson_by_z_by_fly'] = model.pearson_by_z_by_fly(emissions, soft_emission_predictions_per_state, z_probs)
        # model_ckp[data_key][f'{prefix}_pearson_by_z_and_o_soft'] = model.pearson_by_z_by_o(emissions, soft_emission_predictions_per_state, z_probs)
        model_ckp[data_key][f'{prefix}_pearson_by_z_and_o_by_fly'] = model.pearson_by_z_and_o_by_fly(emissions, soft_emission_predictions_per_state, z_probs)
        # model_ckp[data_key][f'{prefix}_correlation_by_o_soft'] = model.pearson_by_o(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix}_correlation_by_o_by_fly_soft'] = model.pearson_by_o_by_fly(emissions, soft_emission_predictions)

        # max_c, max_lags = model.correlation_max_by_o(emissions, soft_emission_predictions)
        # model_ckp[data_key][f'{prefix}_correlation_max_by_o_soft'] = max_c
        # model_ckp[data_key][f'{prefix}_correlation_max_lags_by_o_soft'] = max_lags

        max_c, max_lags = model.correlation_max_by_o_by_fly(emissions, soft_emission_predictions)
        model_ckp[data_key][f'{prefix}_correlation_max_by_o_by_fly_soft'] = max_c
        model_ckp[data_key][f'{prefix}_correlation_max_lags_by_o_by_fly_soft'] = max_lags
        return

    evaluate('train')
    evaluate('test')
    if output_dir:
        joblib.dump(model_ckp, os.path.join(output_dir, 'model.pkl'))
        print("Full checkpoint dumped.")
    return model_ckp


def update_labels(data_config):
    input_labels_text = ({
        'mFV': 'mFV',
        'mLS': 'mLS',
        'fFV': 'fFV',
        'fLS': 'fLS',
        'mfDist': 'mfDist',

        'mFV_directedlr': 'mFV x side',
        'mLS_directedlr': 'mLS x side',
        'fFV_directedlr': 'fFV x side',
        'fLS_directedlr': 'fLS x side',
        'mfDist_directedlr': 'mfDist x side',

        'fmAng_sin': 'sin(fmAng)',
        'fmAng_cos': 'cos(fmAng)',

        'wingAlign': 'wingAng',
        'pfast_i': 'pulse',
        'sine_i': 'sine',
        'pfast_i_directedlr': 'pulse x side',
        'sine_i_directedlr': 'sine x side',

        'tap2': 'tap',
        'tap2_directedlr': 'tap x side',

        # 'fDistWall': 'distWall',
    })
    input_labels_jr_text = ({
        'mFV': 'mFV',
        'mLS': 'mLS',
        'fFV': 'fFV',
        'fLS': 'fLS',
        'mfDist': 'mfDist',

        'mFV_directedlr': 'mFV (s)',
        'mLS_directedlr': 'mLS (s)',
        'fFV_directedlr': 'fFV (s)',
        'fLS_directedlr': 'fLS (s)',
        'mfDist_directedlr': 'mfDist (s)',

        'fmAng_sin': 'sin(fmAng)',
        'fmAng_cos': 'cos(fmAng)',

        'wingAlign': 'wingAng',
        'pfast_i': 'pulse',
        'sine_i': 'sine',
        'pfast_i_directedlr': 'pulse (s)',
        'sine_i_directedlr': 'sine (s)',

        'tap2': 'tap',
        'tap2_directedlr': 'tap (s)',

    })
    emission_labels_text = ({
        'fFV': 'forward velocity',
        'fLV': 'lateral velocity',
        'fLS': 'lateral speed',
        'fAV': 'angular velocity',
        'fAS': 'angular speed',
        'mFV': 'forward velocity',
        'mLV': 'lateral velocity',
        'mAV': 'angular velocity',
        # 'dfmAng': 'z-dfmAng',
        # 'wingFlickTheta': 'wingAngFlick',
        # 'wingFlickBin': 'wingFlickBin',
    })
    emission_labels_jr_text = ({
        'fFV': 'forward\nvelocity',
        'fLV': 'lateral\nvelocity',
        'fLS': 'lateral\nspeed',
        'fAV': 'angular\nvelocity',
        'fAS': 'angular\nspeed',
        'mFV': 'forward\nvelocity',
        'mLV': 'lateral\nvelocity',
        'mAV': 'angular\nvelocity',
        # 'dfmAng': 'z-dfmAng',
        # 'wingFlickTheta': 'wingAngFlick',
        # 'wingFlickBin': 'wingFlickBin',
    })
    emission_labels_jr_jr_text = ({
        'fFV': 'forward',
        'fLV': 'lateral',
        'fLS': 'lateral',
        'fAV': 'angular',
        'fAS': 'angular',
        'mFV': 'forward',
        'mLV': 'lateral',
        'mAV': 'angular',
        # 'dfmAng': 'z-dfmAng',
        # 'wingFlickTheta': 'wingAngFlick',
        # 'wingFlickBin': 'wingFlickBin',
    })
    emission_labels_units_text = ({
        'fFV': 'forward velocity\n(mm/s)',
        'fLV': 'lateral velocity\n(mm/s)',
        'fAV': 'angular velocity\n(deg/s)',
        'fLS': 'lateral speed\n(mm/s)',
        'fAS': 'angular speed\n(deg/s)',
        'mFV': 'forward velocity\n(mm/s)',
        'mLV': 'lateral velocity\n(mm/s)',
        'mAV': 'angular velocity\n(deg/s)',
        # 'dfmAng': 'z-dfmAng',
        # 'wingFlickTheta': 'wingAngFlick',
        # 'wingFlickBin': 'wingFlickBin',
    })
    emission_labels_zscored_text = ({
        'fFV': 'forward velocity\n(zscored)',
        'fLV': 'lateral velocity\n(zscored)',
        'fAV': 'angular velocity\n(zscored)',
        'fLS': 'lateral speed\n(zscored)',
        'fAS': 'angular speed\n(zscored)',
        'mFV': 'forward velocity\n(zscored)',
        'mLV': 'lateral velocity\n(zscored)',
        'mAV': 'angular velocity\n(zscored)',
        # 'dfmAng': 'z-dfmAng',
        # 'wingFlickTheta': 'wingAngFlick',
        # 'wingFlickBin': 'wingFlickBin',
    })
    auxiliary_labels_text = ({
        'mFV': 'mFV',
        'mLS': 'mLS',
        'fFV': 'fFV',
        'fLS': 'fLS',
        'mfDist': 'mfDist',
        'pfast_i': 'pulse',
        'sine_i': 'sine',
        'tap2': 'tap',
        'fmAng_cos': 'front \u2194 back',
        'fmAng_sin': 'right \u2194 left',
    })
    auxiliary_labels_full_text = ({
        'mFV': 'male forward velocity',
        'mLS': 'male lateral speed',
        'fFV': 'female forward velocity',
        'fLS': 'female lateral speed',
        'mfDist': 'distance',
        'pfast_i': 'pulse song',
        'sine_i': 'sine song',
        'tap2': 'tap',
        'fmAng_cos': 'male positioned behind',
        'fmAng_sin': 'male lateral position',
    })
    auxiliary_labels_jr_text = ({
        'mFV': 'mFV',
        'mLS': 'mLS',
        'fFV': 'fFV',
        'fLS': 'fLS',
        'mfDist': 'mfDist',
        'pfast_i': 'pulse',
        'sine_i': 'sine',
        'tap2': 'tap',
        'fmAng_cos': 'cos(fmAng)',
        'fmAng_sin': 'sin(fmAng)',
    })
    auxiliary_emission_labels_text = ({
        'wingFlickBin': 'wing_flick',
        # 'wingFlick2': 'wing_flick2',
    })
    directional_variables = ({
        # 'fmAng_sin': '|male lateral position|',
        'fLV': '|lateral velocity|',
        'fAV': '|angular velocity|',
        'mLV': '|lateral velocity|',
        'mAV': '|angular velocity|',
    })

    # replace labels
    data_config['input_labels'] = {_: input_labels_text[_] for _ in data_config['input_labels_list']}
    data_config['input_labels_jr'] = {_: input_labels_jr_text[_] for _ in data_config['input_labels_list']}

    data_config['auxiliary_labels'] = {_: auxiliary_labels_text[_] for _ in data_config['auxiliary_labels_list']}
    data_config['auxiliary_labels_jr'] = {_: auxiliary_labels_jr_text[_] for _ in data_config['auxiliary_labels_list']}
    data_config['auxiliary_labels_full'] = {_: auxiliary_labels_full_text[_] for _ in data_config['auxiliary_labels_list']}

    data_config['emission_labels_units'] = data_config['emission_labels'].copy()
    data_config['emission_labels_zscored'] = data_config['emission_labels'].copy()
    data_config['emission_labels_jr'] = data_config['emission_labels'].copy()
    data_config['emission_labels_jr_jr'] = data_config['emission_labels'].copy()
    data_config['emission_labels_dict'] = data_config['emission_labels'].copy()
    data_config['emission_labels_dict'].update({k: v for k, v in emission_labels_text.items() if k in data_config['emission_labels']})
    data_config['emission_labels_jr'].update({k: v for k, v in emission_labels_jr_text.items() if k in data_config['emission_labels_jr']})
    data_config['emission_labels_jr_jr'].update({k: v for k, v in emission_labels_jr_jr_text.items() if k in data_config['emission_labels_jr_jr']})
    data_config['emission_labels_units'].update({k: v for k, v in emission_labels_units_text.items() if k in data_config['emission_labels_units']})
    data_config['emission_labels_zscored'].update({k: v for k, v in emission_labels_zscored_text.items() if k in data_config['emission_labels_zscored']})
    data_config['auxiliary_emission_labels'].update({k: v for k, v in auxiliary_emission_labels_text.items() if k in data_config['auxiliary_emission_labels']})
    data_config['directional_variables'] = directional_variables

    return


def generate_figures(model_dir, savefig=True, display=False, override_fig_dir=True):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    update_labels(data_config)

    fig_dir = os.path.join(model_dir, 'figures')
    if os.path.exists(fig_dir) and override_fig_dir:
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    # supp_fig_dir = os.path.join(fig_dir, 'supp_figures')
    scores_fig_dir = os.path.join(fig_dir, 'scores_figures')
    filters_fig_dir = os.path.join(fig_dir, 'filters_figures')
    dists_fig_dir = os.path.join(fig_dir, 'dists_figures')
    # os.makedirs(supp_fig_dir, exist_ok=True)
    os.makedirs(scores_fig_dir, exist_ok=True)
    os.makedirs(filters_fig_dir, exist_ok=True)
    os.makedirs(dists_fig_dir, exist_ok=True)

    learned_params = model_ckp['learned_params']
    learned_lps = model_ckp['learned_lps']
    emission_labels = data_config['emission_labels']
    emission_labels_dict = data_config['emission_labels_dict']
    print("emission_labels", emission_labels)
    print("emission_labels_dict", emission_labels_dict)
    input_labels_jr = data_config['input_labels_jr']
    emission_labels_jr = data_config['emission_labels_jr']
    emission_labels_jr_jr = data_config['emission_labels_jr_jr']
    emission_labels_units = data_config['emission_labels_units']
    emission_labels_zscored = data_config['emission_labels_zscored']
    auxiliary_labels = data_config['auxiliary_labels']
    auxiliary_labels_jr = data_config['auxiliary_labels_jr']
    auxiliary_labels_full = data_config['auxiliary_labels_full']
    auxiliary_emission_labels = data_config['auxiliary_emission_labels']
    directional_variables = data_config['directional_variables']
    effective_fps = data_config['effective_fps']
    num_states = model_ckp['num_states']
    model_prefix = model_ckp['prefix']

    if 'hmm' in model_prefix or 'HMM' in model_prefix:
        plots.plot_loss(learned_lps, savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_expected_occupancy(calculate_steady_state_p(learned_params.transitions.transition_matrix),
                                savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_empirical_occupancy(model_ckp['test_data']['test_stateseq'], model_config,
            title='[Test Data]', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_empirical_occupancy([*model_ckp['train_data']['train_stateseq'], *model_ckp['test_data']['test_stateseq']], model_config,
                                title='', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_ethogram(learned_params.transitions.transition_matrix,
                            savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_transition_matrix(learned_params.transitions.transition_matrix,
                                     savefig=savefig, fig_dir=fig_dir, display=display)
        z_seqs = [*model_ckp['train_data']['train_stateseq'], *model_ckp['test_data']['test_stateseq']]
        plots.plot_state_dwell_times(calc_dwell_times_by_z(z_seqs, num_states), num_states, effective_fps,
                                     savefig=savefig, fig_dir=dists_fig_dir, display=display)

        padded_arrays, n_le = pad_to_equal_length([*model_ckp['train_data']['train_state_probs'], *model_ckp['test_data']['test_state_probs']])
        plots.plot_prob_states_aligned(padded_arrays, n_le, 300, model_config, title=f'All data',
                                       xticks=['0', '30'], xlabel='Time (min)', savefig=savefig, fig_dir=dists_fig_dir, display=display)

        train_nocop_state_probs = [model_ckp['train_data']['train_state_probs'][i] for i, c in enumerate(model_ckp['train_data']['train_copulation_bools']) if c == False]
        test_nocop_state_probs = [model_ckp['test_data']['test_state_probs'][i] for i, c in enumerate(model_ckp['test_data']['test_copulation_bools']) if c == False]
        nocop_state_probs = [*train_nocop_state_probs, *test_nocop_state_probs]
        padded_arrays, n_le = pad_to_equal_length(nocop_state_probs)
        plots.plot_prob_states_aligned(padded_arrays, n_le, 300, config=model_config, title=f'All No Copulation data',
                                       xticks=['0', '30'], xlabel='Time (min)', savefig=savefig, fig_dir=dists_fig_dir, display=display)

        train_cop_state_probs = [model_ckp['train_data']['train_state_probs'][i] for i, c in enumerate(model_ckp['train_data']['train_copulation_bools']) if c == True]
        test_cop_state_probs = [model_ckp['test_data']['test_state_probs'][i] for i, c in enumerate(model_ckp['test_data']['test_copulation_bools']) if c == True]
        cop_state_probs = [*train_cop_state_probs, *test_cop_state_probs]
        plots.plot_prob_states_aligned(normalize_to_equal_length(cop_state_probs, GRID=50000), None, 300,
                                       config=model_config, title=f'All Copulation data',
                                       xticks=['Start', 'Copulation'], xlabel='Time (in courtship)', savefig=savefig, fig_dir=dists_fig_dir, display=display)

    if 'hmm' in model_prefix or 'HMM' in model_prefix:
        weights = learned_params.emissions.weights
    elif 'logrhmm' in model_prefix:
        weights = learned_params.emissions.weights[:, np.newaxis, :]
    elif model_prefix == 'lr':
        weights = learned_params['w']
    else:
        raise Exception(f'wrong prefix={model_prefix}')

    def plot_func(prefix):
        data_key = f'{prefix}_data'
        # emissions = model_ckp[data_key][f'{prefix}_emissions']
        # inputs = model_ckp[data_key][f'{prefix}_inputs']
        # aux_data = model_ckp[data_key][f'{prefix}_aux_data']
        # aux_emissions = model_ckp[data_key][f'{prefix}_aux_emissions']
        # stateseq = model_ckp[data_key][f'{prefix}_stateseq']
        # output_mn_std = model_ckp[data_key][f'{prefix}_output_mn_std']

        # plots.plot_var_explained_by_z(model_ckp[data_key][f'{prefix}_score_by_z_soft'], title=f'{prefix} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z_by_fly(model_ckp[data_key][f'{prefix}_score_by_z_by_fly_soft'], title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)
        # plots.plot_var_explained_by_o(model_ckp[data_key][f'{prefix}_score_by_o_soft'], emission_labels_jr, title=f'{prefix} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_o_by_fly(model_ckp[data_key][f'{prefix}_score_by_o_by_fly_soft'], emission_labels_jr, title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)
        # plots.plot_var_explained_by_z_o(model_ckp[data_key][f'{prefix}_score_by_z_and_o_soft'], emission_labels_jr, title=f'{prefix} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_var_explained_by_z_o_by_fly(model_ckp[data_key][f'{prefix}_score_by_z_and_o_by_fly_soft'], emission_labels_jr, title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)

        # plots.plot_pearson_by_z(model_ckp[data_key][f'{prefix}_pearson_by_z'], title=f'{prefix} data', savefig=savefig, fig_dir=fig_dir, display=display)
        # plots.plot_pearson_by_z_vs_all(model_ckp[data_key][f'{prefix}_pearson'], model_ckp[data_key][f'{prefix}_pearson_by_z'], title=f'{prefix} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_pearson_by_z_by_fly(model_ckp[data_key][f'{prefix}_pearson_by_z_by_fly'], title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)
        plots.plot_pearson_by_z_by_fly_vs_all(model_ckp[data_key][f'{prefix}_pearson_by_fly'], model_ckp[data_key][f'{prefix}_pearson_by_z_by_fly'], title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)
        # plots.plot_correlation_by_o(model_ckp[data_key][f'{prefix}_correlation_by_o_soft'], emission_labels_jr, title=f'{prefix} data (lag=0)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_correlation_by_o_by_fly(model_ckp[data_key][f'{prefix}_correlation_by_o_by_fly_soft'], emission_labels_jr, title=f'{prefix} data (lag=0)', savefig=savefig, fig_dir=scores_fig_dir, display=display)
        # plots.plot_pearson_by_z_o(model_ckp[data_key][f'{prefix}_pearson_by_z_and_o_soft'], emission_labels_jr, title=f'{prefix} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_pearson_by_z_o_by_fly(model_ckp[data_key][f'{prefix}_pearson_by_z_and_o_by_fly'], emission_labels_jr, title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)

        # plots.plot_correlation_by_o(model_ckp[data_key][f'{prefix}_correlation_max_by_o_soft'], emission_labels_jr, title=f'{prefix} data (max lag)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_correlation_by_o_by_fly(model_ckp[data_key][f'{prefix}_correlation_max_by_o_by_fly_soft'], emission_labels_jr, title=f'{prefix} data (max lag)', savefig=savefig, fig_dir=scores_fig_dir, display=display)

        # plots.plot_correlation_lags_by_o(model_ckp[data_key][f'{prefix}_correlation_max_lags_by_o_soft'], emission_labels_jr, effective_fps, title=f'{prefix} data (max lag)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_correlation_lags_by_o_by_fly(model_ckp[data_key][f'{prefix}_correlation_max_lags_by_o_by_fly_soft'], emission_labels_jr, effective_fps, title=f'{prefix} data (max lag)', savefig=savefig, fig_dir=scores_fig_dir, display=display)

        # plots.plot_auxem_acc_by_z_o(stateseq, aux_emissions, model_ckp[data_key][f'{prefix}_probs_z_and_auxo'], model_config, auxiliary_emission_labels, skip_states=[0], title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)

        return

    plot_func('train')
    plot_func('test')

    # overall scores
    plots.plot_var_explained(model_ckp['train_data']['train_score'], model_ckp['test_data']['test_score'], savefig=savefig, fig_dir=scores_fig_dir, display=display)
    plots.plot_var_explained_by_fly(model_ckp['train_data']['train_score_by_fly'], model_ckp['test_data']['test_score_by_fly'], savefig=savefig, fig_dir=scores_fig_dir, display=display)
    plots.plot_pearson(model_ckp['train_data']['train_pearson'], model_ckp['test_data']['test_pearson'], savefig=savefig, fig_dir=scores_fig_dir, display=display)
    plots.plot_pearson_by_fly(model_ckp['train_data']['train_pearson_by_fly'], model_ckp['test_data']['test_pearson_by_fly'], savefig=savefig, fig_dir=scores_fig_dir, display=display)
    plots.plot_ll(model_ckp['train_data']['train_lp'], model_ckp['test_data']['test_lp'], data_config, savefig=savefig, fig_dir=scores_fig_dir, display=display)
    plots.plot_ll_by_fly(model_ckp['train_data']['train_lps_by_fly'], model_ckp['test_data']['test_lps_by_fly'], data_config, savefig=savefig, fig_dir=scores_fig_dir, display=display)

    # plots common to train and test
    plots.plot_legends(num_states, data_config, savefig=savefig, fig_dir=fig_dir, display=display)

    all_emissions = [*model_ckp['train_data']['train_emissions'], *model_ckp['test_data']['test_emissions']]
    all_soft_predictions_per_state = [*model_ckp['train_data']['train_soft_predictions_per_state'], *model_ckp['test_data']['test_soft_predictions_per_state']]
    all_stateseq = [*model_ckp['train_data']['train_stateseq'], *model_ckp['test_data']['test_stateseq']]
    all_aux_data = [*model_ckp['train_data']['train_aux_data'], *model_ckp['test_data']['test_aux_data']]
    all_output_mn_std = [*model_ckp['train_data']['train_output_mn_std'], *model_ckp['test_data']['test_output_mn_std']]
    all_aux_mn_std = [*model_ckp['train_data']['train_aux_mn_std'], *model_ckp['test_data']['test_aux_mn_std']]

    plots.plot_state_aux_dists_reformatted(
        get_aux_by_state(all_aux_data, all_stateseq, num_states, all_aux_mn_std, rescaled=True,),
        auxiliary_labels, data_config, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_aux_sorted_odists(get_emissions_by_state(all_aux_data, all_stateseq, num_states, rescaled=False),
        get_emissions_by_state(all_emissions, all_stateseq, num_states, rescaled=False),
        auxiliary_labels_full, emission_labels_jr_jr, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_o_dists(
        get_emissions_by_state(all_emissions, all_stateseq, num_states, rescaled=False),
        emission_labels_zscored, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_o_dists_reformatted(
        get_emissions_by_state(all_emissions, all_stateseq, num_states, all_output_mn_std, rescaled=True, effective_fps=effective_fps),
        emission_labels_units, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_aux_dists(
        get_aux_by_state(all_aux_data, all_stateseq, num_states, rescaled=False),
        auxiliary_labels, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_aux_o_mean(
        get_emissions_by_state(all_aux_data, all_stateseq, num_states, rescaled=False),
        get_emissions_by_state(all_emissions, all_stateseq, num_states, rescaled=False),
        auxiliary_labels_jr, emission_labels_jr_jr, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_aux_sorted_mean_o(get_emissions_by_state(all_aux_data, all_stateseq, num_states, rescaled=False),
        get_emissions_by_state(all_emissions, all_stateseq, num_states, rescaled=False),
        auxiliary_labels_full, emission_labels_dict, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_aux_sorted_o_mean_directional(get_emissions_by_state(all_aux_data, all_stateseq, num_states, rescaled=False),
        get_emissions_by_state(all_emissions, all_stateseq, num_states, rescaled=False),
        auxiliary_labels_full, emission_labels_dict, directional_variables, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_o_dists_otherfilters(all_soft_predictions_per_state, all_stateseq, num_states, emission_labels_zscored,
                                          title='all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)

    # plot filters for regular emissions
    for skip_states in [[], [0]]:
        if num_states <= 1 and skip_states:
            continue    # skip skip_states if there's only state
        plots.plot_filters(weights, data_config, emission_labels, filesuffix='emissions', skip_states=skip_states, savefig=savefig, fig_dir=filters_fig_dir, display=display)
        plots.plot_filters(weights, data_config, emission_labels, filesuffix='emissions', skip_states=skip_states, sharey='row', savefig=savefig, fig_dir=filters_fig_dir, display=display)
        plots.plot_filters_separate_emissions(weights, data_config, emission_labels, filesuffix='emissions', sharey='row', skip_states=skip_states, savefig=savefig, fig_dir=filters_fig_dir, display=display)
        plots.plot_filters_separate_emissions(weights, data_config, emission_labels, filesuffix='emissions', sharey='row', skip_states=skip_states, saveindividual=True, savefig=savefig, fig_dir=filters_fig_dir, display=display)
        plots.plot_filter_amplitudes(weights, data_config, prefix='emissions', skip_states=skip_states, savefig=savefig, fig_dir=filters_fig_dir, display=display)
        plots.plot_filter_amplitudes(weights, data_config, prefix='emissions', plot_top_k=5, skip_states=skip_states, savefig=savefig, fig_dir=filters_fig_dir, display=display)
        plots.plot_filters_statewise(weights, data_config, prefix='emissions', only_plot_inputs=['mFV', 'pfast_i', 'sine_i', 'tap2'], skip_states=skip_states, savefig=savefig, fig_dir=filters_fig_dir, display=display)

    # # plot filters for other emissions
    # # TODO broken
    # auxem_weights = model_ckp['logreg_params']['w']
    # plots.plot_filters_newbasis(auxem_weights, data_config, auxiliary_emission_labels, filesuffix='aux_emissions', savefig=savefig, fig_dir=filters_fig_dir, display=display)
    # plots.plot_filters_newbasis(auxem_weights, data_config, auxiliary_emission_labels, filesuffix='aux_emissions', sharey='row', savefig=savefig, fig_dir=filters_fig_dir, display=display)
    # plots.plot_filter_amplitudes(auxem_weights, data_config, prefix='aux_emissions', savefig=savefig, fig_dir=filters_fig_dir, display=display)
    # plots.plot_filter_amplitudes(auxem_weights, data_config, prefix='aux_emissions', plot_top_k=min(5, len(auxiliary_labels)), savefig=savefig, fig_dir=filters_fig_dir, display=display)
    # plots.plot_filters_statewise(auxem_weights, data_config, prefix='aux_emissions', skip_states=[], only_plot_inputs=['mFV', 'pfast_i', 'sine_i', 'tap2'], savefig=savefig, fig_dir=filters_fig_dir, display=display)
    #
    # if num_states > 1:
    #     plots.plot_filters_newbasis(auxem_weights, data_config, auxiliary_emission_labels, filesuffix='aux_emissions', skip_states=[0], savefig=savefig, fig_dir=filters_fig_dir, display=display)
    #     plots.plot_filters_newbasis(auxem_weights, data_config, auxiliary_emission_labels, filesuffix='aux_emissions', sharey='row', skip_states=[0], savefig=savefig, fig_dir=filters_fig_dir, display=display)
    #     plots.plot_filter_amplitudes(auxem_weights, data_config, prefix='aux_emissions', skip_states=[0], savefig=savefig, fig_dir=filters_fig_dir, display=display)
    #     plots.plot_filter_amplitudes(auxem_weights, data_config, prefix='aux_emissions', skip_states=[0], plot_top_k=min(5, len(auxiliary_labels)), savefig=savefig, fig_dir=filters_fig_dir, display=display)
    #     plots.plot_filters_statewise(auxem_weights, data_config, prefix='aux_emissions', skip_states=[0], only_plot_inputs=['mFV', 'pfast_i', 'sine_i', 'tap2'], savefig=savefig, fig_dir=filters_fig_dir, display=display)
    return


def plot_xlims(model_dir, windows, batch, prefix, trajs_dir, trajs2d_dir, probs_dir, suffix='', savefig=True, display=False, gen_corr_video=False):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    update_labels(data_config)

    data_key = f'{prefix}_data'
    sessions_key = f'{prefix}_session_keys'
    dwnsmpl_key = f'{prefix}_downsampled_indices'

    effective_fps = data_config['effective_fps']
    key_b = model_ckp[data_key][sessions_key][batch]

    for i, xlim_ in enumerate(windows):
        xlim = (int(xlim_[0]), int(xlim_[1]))
        len_traj = xlim[1] - xlim[0]
        xlim_orig = (int(model_ckp[data_key][dwnsmpl_key][batch][xlim[0]]), int(model_ckp[data_key][dwnsmpl_key][batch][xlim[1]]))
        plots.plot_smoothed_probs(model_ckp[data_key][f'{prefix}_state_probs'], model_config, data_config, batch, effective_fps, xlim=xlim, xlim_orig=xlim_orig, prefix=prefix, suffix=suffix, savefig=savefig, fig_path=f'{probs_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)
            # plots.plot_comparison_probs(model_ckp[data_key][f'{prefix}_state_probs'], model_ckp[data_key][f'{prefix_data}_fwd_state_probs'], model_config, batch, effective_fps, xlim=xlim, xlim_orig=xlim_orig, prefix_data=prefix_data, suffix=suffix, savefig=savefig, fig_path=f'{probs_dir}/{prefix}{batch}_xlim={xlim}{suffix}_.pdf', display=display)
        # plots.plot_trajectories(model_ckp, model_config, data_config, batch, states_in_bgr=True, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)
        plots.plot_trajectories(model_ckp, model_config, data_config, batch, states_in_bgr=False, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)
        plots.plot_trajectories_statewise(model_ckp, model_config, data_config, batch, states_in_bgr=False, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}_perstate{suffix}', display=display)
        # plots.plot_trajectories2D(model_ckp, model_config, data_config, batch, states_in_bgr=True, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs2d_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)
        # plots.plot_trajectories_w_partner(model_ckp, model_config, data_config, batch, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs_dir}/{prefix}{batch}_w_partner_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)

        if gen_corr_video:
            clip_session(os.path.join('/Volumes/murthy/usingla/gold_dataset/wt/mp4', key_b.replace(".h5", ".mp4")), xlim_orig, output_path=f'{trajs_dir}/{prefix}{batch}_xlim_orig={xlim_orig}_xlim={xlim}{suffix}.mp4')
        break

    return


def generate_state_clips(model_dir, savefig=True, display=False, gen_corr_video=False):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    update_labels(data_config)
    # effective_fps = data_config['effective_fps']

    fig_dir = os.path.join(model_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    def f(prefix):

        data_key = f'{prefix}_data'
        n_sessions = len(model_ckp[data_key][f'{prefix}_session_keys'])

        for batch in np.random.choice(n_sessions, size=min(5, n_sessions)):
            key_b = model_ckp[data_key][f'{prefix}_session_keys'][batch]
            stateseq = model_ckp[data_key][f'{prefix}_stateseq'][batch]
            # num_timestamps = stateseq.shape[0]
            print("batch", batch, "key_b", key_b)

            for z in range(model_config['num_states']):
                windows = get_state_indices(stateseq, z, min_length=10, max_clips=10)
                print(z, windows)

                state_trajs_dir = os.path.join(fig_dir, 'state_trajs', str(z+1))
                state_trajs2d_dir = os.path.join(fig_dir, 'state_trajs2d', str(z+1))
                state_probs_dir = os.path.join(fig_dir, 'state_probs', str(z+1))
                os.makedirs(state_trajs_dir, exist_ok=True)
                os.makedirs(state_trajs2d_dir, exist_ok=True)
                os.makedirs(state_probs_dir, exist_ok=True)
                plot_xlims(model_dir, windows, batch, prefix, state_trajs_dir, state_trajs2d_dir, state_probs_dir, savefig=savefig, display=display, gen_corr_video=gen_corr_video)
            # break

    f('train')
    return


def generate_state_traces(model_dir, dataset='wt', savefig=True, display=False):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    update_labels(data_config)
    # effective_fps = data_config['effective_fps']

    state_traces_dir = os.path.join(model_dir, 'figures/state_traces')
    if os.path.exists(state_traces_dir):
        shutil.rmtree(state_traces_dir)
    os.makedirs(state_traces_dir, exist_ok=True)

    if dataset == 'wt':
        ftracks = joblib.load('processed_ftracks.pkl')
        mtracks = joblib.load('processed_mtracks.pkl')
        dataset_basepath = '/Volumes/murthy/usingla/gold_dataset/wt/h5'
    elif dataset == 'wt_fred':
        ftracks = joblib.load('wt_fredcleaned_processed_ftracks.pkl')
        mtracks = joblib.load('wt_fredcleaned_processed_mtracks.pkl')
        dataset_basepath = '/Volumes/fileset-mmurthy/usingla/fred_data/'

    print(ftracks.keys())

    def f(prefix):

        data_key = f'{prefix}_data'
        n_sessions = len(model_ckp[data_key][f'{prefix}_session_keys'])
        dwnsmpl_key = f'{prefix}_downsampled_indices'

        for batch in np.random.choice(n_sessions, size=min(30, n_sessions)):
            key_b = model_ckp[data_key][f'{prefix}_session_keys'][batch]
            stateseq = model_ckp[data_key][f'{prefix}_stateseq'][batch]
            print("batch", batch, "key_b", key_b)

            for z in range(model_config['num_states']):
                windows = get_state_indices(stateseq, z, min_length=10, max_length=20, max_clips=9)

                if len(windows) < 9:
                    print(f"{batch}: {key_b}, not enough state {z+1} clips.")
                    continue

                windows_orig = []
                for i, xlim_ in enumerate(windows):
                    xlim = (int(xlim_[0]), int(xlim_[1]))
                    xlim_orig = (int(model_ckp[data_key][dwnsmpl_key][batch][xlim[0]]), int(model_ckp[data_key][dwnsmpl_key][batch][xlim[1]]))
                    windows_orig.append(xlim_orig)

                if dataset == 'wt':
                    expt_path = os.path.join(dataset_basepath, key_b)
                elif dataset == 'wt_fred':
                    expt_path = os.path.join(dataset_basepath, key_b, key_b.split('_')[-1], '000000.mp4.inference.000_000000.analysis.h5')

                if expt_path not in ftracks:
                    print("not processed.")
                    return

                fTrx = ftracks[expt_path]
                mTrx = mtracks[expt_path]

                plots.plot_traces_session(fTrx, mTrx,
                                          windows_orig, z=z,
                                          output_path=os.path.join(state_traces_dir, str(z+1), f'{prefix}{batch}.pdf'),
                                          title=f'State {z+1}', savefig=savefig, display=display)

    f('train')
    f('test')
    return


def generate_trajs(model_dir, savefig=True, display=False, gen_corr_video=False):

    model_ckp, data_config, _ = load_specific_path(model_dir)
    if model_ckp is None:
        return
    # print(data_config['auxiliary_labels'])
    update_labels(data_config)
    # print(data_config['auxiliary_labels'])
    # print(data_config['emission_labels_zscored'])
    effective_fps = data_config['effective_fps']

    fig_dir = os.path.join(model_dir, 'figures')
    trajs_dir = os.path.join(fig_dir, 'trajs')
    trajs2d_dir = os.path.join(fig_dir, 'trajs2d')
    probs_dir = os.path.join(fig_dir, 'probs')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(trajs_dir, exist_ok=True)
    os.makedirs(trajs2d_dir, exist_ok=True)
    os.makedirs(probs_dir, exist_ok=True)

    def f(prefix):

        data_key = f'{prefix}_data'
        n_sessions = len(model_ckp[data_key][f'{prefix}_session_keys'])

        for batch in np.random.choice(n_sessions, size=min(10, n_sessions)):
            # batch = 24 if prefix == 'train' else batch
            # batch = 10 if prefix == 'test' else batch
            key_b = model_ckp[data_key][f'{prefix}_session_keys'][batch]
            num_timestamps = model_ckp[data_key][f'{prefix}_stateseq'][batch].shape[0]
            print("batch", batch, "key_b", key_b, "num_timestamps", num_timestamps)
            windows = get_windows_to_plot(effective_fps, num_timestamps)
            # print("windows", windows)
            plot_xlims(model_dir, windows, batch, prefix, trajs_dir, trajs2d_dir, probs_dir, savefig=savefig, display=display, gen_corr_video=gen_corr_video)
            lastwindows = get_cop_window_to_plot(data_config['effective_fps'], num_timestamps)
            plot_xlims(model_dir, lastwindows, batch, prefix, trajs_dir, trajs2d_dir, probs_dir, suffix='(final 30 seconds)', savefig=savefig, display=display, gen_corr_video=gen_corr_video)
            fullwindows = get_full_window_to_plot(data_config['effective_fps'], num_timestamps)
            plot_xlims(model_dir, fullwindows, batch, prefix, trajs_dir, trajs2d_dir, probs_dir, suffix='(whole session)', savefig=savefig, display=display, gen_corr_video=gen_corr_video)
    f('train')
    f('test')
    return


def generate_TAs(model_dir, savefig=True, display=False):
    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return
    fig_dir = os.path.join(model_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print(data_config['auxiliary_labels_list'])
    update_labels(data_config)

    # plots.plot_STAs(model_ckp, model_config, data_config, prefix='train', savefig=savefig, display=display)
    # plots.plot_STAs(model_ckp, model_config, data_config, prefix='test', savefig=savefig, display=display)
    # plots.plot_ETSPs(model_ckp, model_config, data_config, savefig=savefig, display=display)
    # plots.plot_ETAs(model_ckp, model_config, data_config, fig_dir=fig_dir, savefig=savefig, display=display)
    plots.plot_ETAs_all(model_ckp, data_config, fig_dir=fig_dir, savefig=savefig, display=display)
    return



# def generate_videos(model_dir, override_vid_dir=True):
#
#     model_ckp, data_config, model_config = load_specific_path(model_dir)
#     if model_ckp is None:
#         return
#
#     update_labels(data_config)
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
