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
from utilities.logreg import train_logreg_aux_emissions


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

    print("model.learned_params", model.learned_params)

    print("Calculating train and test logprobs...")
    train_lp = model.get_data_logprob(train_emissions, train_inputs)
    train_lps_by_fly = model.get_data_logprob_by_fly(train_emissions, train_inputs)
    test_lp = model.get_data_logprob(test_emissions, test_inputs)
    test_lps_by_fly = model.get_data_logprob_by_fly(test_emissions, test_inputs)
    print("Train logprob:", train_lp, "Test logprob:", test_lp)

    model_ckp = {
        'prefix': model.prefix,
        'model': model if model.prefix not in ['chance'] else '',     # chance model cannot unpickle tfd distribution
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

        soft_emission_predictions, z_seqs, soft_emission_predictions_per_state, z_probs, fwd_z_probs = model.predict(emissions, inputs)
        # z_probs, fwd_z_probs = model.get_state_probs(emissions, inputs)

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
        model_ckp[data_key][f'{prefix}_correlation_max_by_o_by_fly_soft'] = max_c   # This is pearson_by_o
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

        'mFV_directedlr2': 'mFV x side',
        'mLS_directedlr2': 'mLS x side',
        'fFV_directedlr2': 'fFV x side',
        'fLS_directedlr2': 'fLS x side',
        'mfDist_directedlr2': 'mfDist x side',

        'fmAng_sin': 'sin(fmAng)',
        'fmAng_cos': 'cos(fmAng)',

        'mfAng_sin': 'sin(mfAng)',
        'mfAng_cos': 'cos(mfAng)',

        'wingAlign': 'wingAng',
        'wingAlign_song_i_directedlr2': 'wing x side',
        'pfast_i': 'pulse',
        'pulse_i': 'pulse',
        'sine_i': 'sine',
        'pfast_i_directedlr': 'pulse x side',
        'pulse_i_directedlr2': 'pulse x side',
        'sine_i_directedlr': 'sine x side',
        'sine_i_directedlr2': 'sine x side',

        'tap2': 'tap',
        'tap2_directedlr': 'tap x side',
        'tap2_directedlr2': 'tap x side',

        # 'fDistWall': 'distWall',
    })
    input_labels_jr_text = ({
        'mFV': 'mFV',
        'mLS': 'mLS',
        'fFV': 'fFV',
        'fLS': 'fLS',
        'mfDist': 'mfDist',

        'mFV_directedlr2': 'mFV (s)',
        'mLS_directedlr2': 'mLS (s)',
        'fFV_directedlr2': 'fFV (s)',
        'fLS_directedlr2': 'fLS (s)',
        'mfDist_directedlr2': 'mfDist (s)',

        'fmAng_sin': 'sin(fmAng)',
        'fmAng_cos': 'cos(fmAng)',

        'mfAng_sin': 'sin(mfAng)',
        'mfAng_cos': 'cos(mfAng)',

        'wingAlign': 'wingAng',
        'wingAlign_song_i_directedlr2': 'wing (s)',
        'pfast_i': 'pulse',
        'pulse_i': 'pulse',
        'sine_i': 'sine',
        'pfast_i_directedlr': 'pulse (s)',
        'pulse_i_directedlr2': 'pulse (s)',
        'sine_i_directedlr': 'sine (s)',
        'sine_i_directedlr2': 'sine (s)',

        'tap2': 'tap',
        'tap2_directedlr': 'tap (s)',
        'tap2_directedlr2': 'tap (s)',

    })
    emission_labels_text = ({
        'fFV': 'forward velocity',
        'fFA': 'forward acc',
        'fLV': 'lateral velocity',
        'fLA': 'lateral acc',
        'fLS': 'lateral speed',
        'fAV': 'angular velocity',
        'fAA': 'angular acc',
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
        'fFA': 'forward\nacc',
        'fLV': 'lateral\nvelocity',
        'fLA': 'lateral\nacc',
        'fLS': 'lateral\nspeed',
        'fAV': 'angular\nvelocity',
        'fAA': 'angular\nacc',
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
        'fFA': 'forward-a',
        'fLV': 'lateral',
        'fLA': 'lateral-a',
        'fLS': 'lateral',
        'fAV': 'angular',
        'fAA': 'angular-a',
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
        'fFA': 'forward acc\n(mm/s2)',
        'fLV': 'lateral velocity\n(mm/s)',
        'fLA': 'lateral acc\n(mm/s2)',
        'fAV': 'angular velocity\n(deg/s)',
        'fAA': 'angular acc\n(deg/s2)',
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
        'fFA': 'forward acc\n(zscored)',
        'fLV': 'lateral velocity\n(zscored)',
        'fLA': 'lateral acc\n(zscored)',
        'fAV': 'angular velocity\n(zscored)',
        'fAA': 'angular acc\n(zscored)',
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
        'pulse_i': 'pulse',
        'sine_i': 'sine',
        'tap2': 'tap',
        'fmAng_cos': 'front \u2194 back',
        'fmAng_sin': 'right \u2194 left',
        'mfAng_cos': 'front \u2194 back',
        'mfAng_sin': 'right \u2194 left',
        'wingAlign': 'wing',
    })
    auxiliary_labels_full_text = ({
        'mFV': 'male forward velocity',
        'mLS': 'male lateral speed',
        'fFV': 'female forward velocity',
        'fLS': 'female lateral speed',
        'mfDist': 'distance',
        'pfast_i': 'pulse song',
        'pulse_i': 'pulse song',
        'sine_i': 'sine song',
        'tap2': 'tap',
        'fmAng_cos': 'male positioned behind',
        'fmAng_sin': 'male lateral position',
        'mfAng_cos': 'female positioned behind',
        'mfAng_sin': 'female lateral position',
        'wingAlign': 'wing align',
    })
    auxiliary_labels_jr_text = ({
        'mFV': 'mFV',
        'mLS': 'mLS',
        'fFV': 'fFV',
        'fLS': 'fLS',
        'mfDist': 'mfDist',
        'pfast_i': 'pulse',
        'pulse_i': 'pulse',
        'sine_i': 'sine',
        'tap2': 'tap',
        'fmAng_cos': 'cos(fmAng)',
        'fmAng_sin': 'sin(fmAng)',
        'mfAng_sin': 'sin(mfAng)',
        'mfAng_cos': 'cos(mfAng)',
        'wingAlign': 'wing align',
    })
    auxiliary_emission_labels_text = ({
        'wingFlickBin': 'wing_flick',
        # 'wingFlick2': 'wing_flick2',
    })
    directional_variables = ({
        # 'fmAng_sin': '|male lateral position|',
        'fLV': '|lateral velocity|',
        'fLA': '|lateral acc|',
        'fAV': '|angular velocity|',
        'fAA': '|angular acc|',
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
    data_config['auxiliary_emission_labels_dict'] = data_config['auxiliary_emission_labels'].copy()
    data_config['auxiliary_emission_labels_dict'].update({k: v for k, v in auxiliary_emission_labels_text.items() if k in data_config['auxiliary_emission_labels']})
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
    dists_fig_dir = os.path.join(fig_dir, 'dists_figures')
    os.makedirs(scores_fig_dir, exist_ok=True)
    os.makedirs(dists_fig_dir, exist_ok=True)

    learned_params = model_ckp['learned_params']
    learned_lps = model_ckp['learned_lps']
    emission_labels_jr = data_config['emission_labels_jr']
    emission_labels_jr_jr = data_config['emission_labels_jr_jr']
    emission_labels_units = data_config['emission_labels_units']
    emission_labels_zscored = data_config['emission_labels_zscored']
    auxiliary_labels = data_config['auxiliary_labels']
    auxiliary_labels_jr = data_config['auxiliary_labels_jr']
    auxiliary_labels_full = data_config['auxiliary_labels_full']
    effective_fps = data_config['effective_fps']
    num_states = model_ckp['num_states']
    model_prefix = model_ckp['prefix']
    print("model_prefix", model_prefix)
    # model = model_ckp['model']

    if 'hmm' in model_prefix or 'HMM' in model_prefix:
        plots.plot_loss(learned_lps, savefig=savefig, fig_dir=fig_dir, display=display)
        # plots.plot_expected_occupancy(calculate_steady_state_p(learned_params.transitions.transition_matrix),
        #                         savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_empirical_occupancy(model_ckp['test_data']['test_stateseq'], model_config,
            title='[Test Data]', savefig=savefig, fig_dir=fig_dir, display=display)

        z_seqs = [*model_ckp['train_data']['train_stateseq'], *model_ckp['test_data']['test_stateseq']]
        plots.plot_empirical_occupancy(z_seqs, model_config, title='', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_state_dwell_times(calc_dwell_times_by_z(z_seqs, num_states), num_states, effective_fps,
                                     savefig=savefig, fig_dir=dists_fig_dir, display=display)

        padded_arrays, n_le = pad_to_equal_length([*model_ckp['train_data']['train_state_probs'], *model_ckp['test_data']['test_state_probs']])
        # print("padded_arrays", padded_arrays.shape)
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

        if 'id' in model_prefix:
            emp_transition_matrix = calculate_empirical_transition_matrix(z_seqs, num_states)
            plots.plot_transition_matrix(emp_transition_matrix, title='empirical_transition_matrix', savefig=savefig, fig_dir=fig_dir, display=display)
            plots.plot_ethogram(emp_transition_matrix,  title='empirical_ethogram', savefig=savefig, fig_dir=fig_dir, display=display)
        else:
            plots.plot_transition_matrix(learned_params.transitions.transition_matrix, savefig=savefig, fig_dir=fig_dir, display=display)
            plots.plot_ethogram(learned_params.transitions.transition_matrix, savefig=savefig, fig_dir=fig_dir, display=display)

    def plot_func(prefix):
        print("prefix", prefix)
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
        # plots.plot_pearson_by_z_by_fly(model_ckp[data_key][f'{prefix}_pearson_by_z_by_fly'], title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)
        plots.plot_pearson_by_z_by_fly_vs_all(model_ckp[data_key][f'{prefix}_pearson_by_fly'], model_ckp[data_key][f'{prefix}_pearson_by_z_by_fly'], title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)
        # plots.plot_correlation_by_o(model_ckp[data_key][f'{prefix}_correlation_by_o_soft'], emission_labels_jr, title=f'{prefix} data (lag=0)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_correlation_by_o_by_fly(model_ckp[data_key][f'{prefix}_correlation_by_o_by_fly_soft'], emission_labels_jr, title=f'{prefix} data (lag=0)', savefig=savefig, fig_dir=scores_fig_dir, display=display)
        # plots.plot_pearson_by_z_o(model_ckp[data_key][f'{prefix}_pearson_by_z_and_o_soft'], emission_labels_jr, title=f'{prefix} data', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_pearson_by_z_o_by_fly(model_ckp[data_key][f'{prefix}_pearson_by_z_and_o_by_fly'], emission_labels_jr, title=f'{prefix} data', savefig=savefig, fig_dir=scores_fig_dir, display=display)

        # plots.plot_correlation_by_o(model_ckp[data_key][f'{prefix}_correlation_max_by_o_soft'], emission_labels_jr, title=f'{prefix} data (max lag)', savefig=savefig, fig_dir=fig_dir, display=display)
        plots.plot_correlation_by_o_by_fly(model_ckp[data_key][f'{prefix}_correlation_max_by_o_by_fly_soft'], emission_labels_jr, title=f'{prefix} data (max lag)', savefig=savefig, fig_dir=scores_fig_dir, display=display)

        # plots.plot_correlation_lags_by_o(model_ckp[data_key][f'{prefix}_correlation_max_lags_by_o_soft'], emission_labels_jr, effective_fps, title=f'{prefix} data (max lag)', savefig=savefig, fig_dir=fig_dir, display=display)
        # plots.plot_correlation_lags_by_o_by_fly(model_ckp[data_key][f'{prefix}_correlation_max_lags_by_o_by_fly_soft'], emission_labels_jr, effective_fps, title=f'{prefix} data (max lag)', savefig=savefig, fig_dir=scores_fig_dir, display=display)

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

    plots.plot_state_aux_o_mean(
        get_emissions_by_state(all_aux_data, all_stateseq, num_states, rescaled=False),
        get_emissions_by_state(all_emissions, all_stateseq, num_states, rescaled=False),
        auxiliary_labels_jr, emission_labels_jr_jr, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir,
        display=display)
    plots.plot_state_o_dists_otherfilters(all_soft_predictions_per_state, all_stateseq, num_states, emission_labels_zscored,
                                          title='all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_aux_dists_reformatted(
        get_aux_by_state(all_aux_data, all_stateseq, num_states, all_aux_mn_std, rescaled=True,),
        auxiliary_labels, data_config, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    plots.plot_state_aux_dists_reformatted(
        get_aux_by_state(all_aux_data, all_stateseq, num_states, all_aux_mn_std, rescaled=True,),
         auxiliary_labels, data_config, exclude_a=['wingAlign', 'fmAng_sin', 'mfAng_sin'], title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
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
    # plots.plot_state_aux_sorted_mean_o(get_emissions_by_state(all_aux_data, all_stateseq, num_states, rescaled=False),
    #     get_emissions_by_state(all_emissions, all_stateseq, num_states, rescaled=False),
    #     auxiliary_labels_full, emission_labels_dict, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    # plots.plot_state_aux_sorted_o_mean_directional(get_emissions_by_state(all_aux_data, all_stateseq, num_states, rescaled=False),
    #     get_emissions_by_state(all_emissions, all_stateseq, num_states, rescaled=False),
    #     auxiliary_labels_full, emission_labels_dict, directional_variables, title=f'all data', savefig=savefig, fig_dir=dists_fig_dir, display=display)
    return


def enhance_auxem(model_dir):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    if model_ckp['prefix'] == 'chance': # Skip auxem predictions etc on the Chance model
        return

    num_states = model_ckp['num_states']
    train_aux_emissions = model_ckp['train_data'][f'train_aux_emissions']
    test_aux_emissions = model_ckp['test_data'][f'test_aux_emissions']
    train_inputs = model_ckp['train_data'][f'train_inputs']
    test_inputs = model_ckp['test_data'][f'test_inputs']
    train_stateseq = model_ckp['train_data'][f'train_stateseq']
    test_stateseq = model_ckp['test_data'][f'test_stateseq']

    auxem_model_ckp = {
        'train_data': {},
        'test_data': {},
    }

    input_mask_by_auxemission = data_config['input_mask_by_auxemission']
    print("input_mask_by_auxemission", input_mask_by_auxemission, len(input_mask_by_auxemission[0]))

    w, b, train_predict_probas, test_predict_probas, train_f1score, test_f1score, train_preds, test_preds, train_true, test_true = train_logreg_aux_emissions(train_inputs, test_inputs, train_aux_emissions, test_aux_emissions, train_stateseq, test_stateseq, num_states, input_mask_by_auxemission)
    auxem_model_ckp['logreg_params'] = {'w': w, 'b': b}
    auxem_model_ckp['input_mask_by_auxemission'] = input_mask_by_auxemission
    auxem_model_ckp['train_data']['train_probs_z_and_auxo'] = train_predict_probas
    auxem_model_ckp['train_data']['train_f1score_z_and_auxo'] = train_f1score
    auxem_model_ckp['train_data']['train_preds_z_and_auxo'] = train_preds
    auxem_model_ckp['train_data']['train_true_z_and_auxo'] = train_true
    auxem_model_ckp['test_data']['test_probs_z_and_auxo'] = test_predict_probas
    auxem_model_ckp['test_data']['test_f1score_z_and_auxo'] = test_f1score
    auxem_model_ckp['test_data']['test_preds_z_and_auxo'] = test_preds
    auxem_model_ckp['test_data']['test_true_z_and_auxo'] = test_true

    if model_dir:
        joblib.dump(auxem_model_ckp, os.path.join(model_dir, 'auxem_model.pkl'))
        print("Aux model dumped.")

    return


def generate_auxem_figures(model_dir, savefig=True, display=False):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    auxem_model_ckp = load_specific_path_auxem(model_dir)
    if (model_ckp is None) or (auxem_model_ckp is None):
        return

    fig_dir = os.path.join(model_dir, 'auxem_figures')
    # auxem_filters_fig_dir = os.path.join(fig_dir, 'auxem_filters_fig_dir')
    os.makedirs(fig_dir, exist_ok=True)
    # os.makedirs(auxem_filters_fig_dir, exist_ok=True)

    train_predict_probas = auxem_model_ckp['train_data']['train_probs_z_and_auxo']
    train_f1score_z_and_auxo = auxem_model_ckp['train_data']['train_f1score_z_and_auxo']
    train_preds_z_and_auxo = auxem_model_ckp['train_data']['train_preds_z_and_auxo']
    train_true_z_and_auxo = auxem_model_ckp['train_data']['train_true_z_and_auxo']
    test_predict_probas = auxem_model_ckp['test_data']['test_probs_z_and_auxo']
    test_f1score_z_and_auxo = auxem_model_ckp['test_data']['test_f1score_z_and_auxo']
    test_preds_z_and_auxo = auxem_model_ckp['test_data']['test_preds_z_and_auxo']
    test_true_z_and_auxo = auxem_model_ckp['test_data']['test_true_z_and_auxo']

    update_labels(data_config)
    auxiliary_emission_labels = data_config['auxiliary_emission_labels']

    plots.plot_auxem_frac_full_precomputed(train_true_z_and_auxo, test_true_z_and_auxo, title=f'all data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_auxem_frac_by_z_o_traintest_precomputed(train_true_z_and_auxo, test_true_z_and_auxo, model_config, auxiliary_emission_labels, skip_states=[], title=f'all data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_auxem_acc_full_precomputed(train_true_z_and_auxo, train_preds_z_and_auxo, test_true_z_and_auxo, test_preds_z_and_auxo, title=f'all data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_auxem_acc_by_z_o_traintest_precomputed(train_f1score_z_and_auxo, test_f1score_z_and_auxo, model_config, auxiliary_emission_labels, skip_states=[], title=f'all data', savefig=savefig, fig_dir=fig_dir, display=display)
    return


def generate_state_filters(model_dir, savefig=True, display=False):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if (model_ckp is None):
        return

    model_prefix = model_ckp['prefix']
    print('model_prefix', model_prefix)
    if model_prefix in ['lr', 'glm-hmm', 'glmhmm_']:
        return

    fig_dir = os.path.join(model_dir, 'figures')
    state_fig_dir = os.path.join(fig_dir, 'state_figures')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(state_fig_dir, exist_ok=True)

    update_labels(data_config)
    input_labels = data_config['input_labels']
    learned_params = model_ckp['learned_params']
    reg_weights = learned_params.transitions.weights

    input_mask_by_statetrans = data_config['input_mask_by_emission'][0]
    input_list = ['mFV', 'mLS', 'mfDist', 'fmAng_cos', 'pulse_i', 'sine_i', 'tap2']

    plots.plot_statetrans_filters_separate(reg_weights, data_config, input_list, input_mask_by_statetrans, input_labels, filesuffix='allstates', sharey=True, savefig=savefig, fig_dir=state_fig_dir, display=display)
    plots.plot_statetrans_filter_amplitudes(reg_weights, data_config, input_list, input_mask_by_statetrans, input_labels, prefix='allstates', savefig=savefig, fig_dir=state_fig_dir, display=display)
    return


def generate_together_figures(model_dir, savefig=True, display=False):
    """Plots, such as filters, that need to be together for emissions and aux-emissions. """

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    auxem_model_ckp = load_specific_path_auxem(model_dir)
    if (model_ckp is None) or (auxem_model_ckp is None):
        return

    fig_dir = os.path.join(model_dir, 'figures')
    together_filters_fig_dir = os.path.join(fig_dir, 'together_filters_fig_dir')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(together_filters_fig_dir, exist_ok=True)

    update_labels(data_config)
    input_labels = data_config['input_labels']
    model_prefix = model_ckp['prefix']
    num_states = model_ckp['num_states']
    animal = data_config['animal']

    print("model_prefix", model_prefix)

    # plot filters for regular+auxem emissions
    input_mask_by_emission = data_config['input_mask_by_emission']
    input_mask_by_auxemission = np.array(auxem_model_ckp['input_mask_by_auxemission'])
    input_mask_by_allemission = np.concatenate((input_mask_by_emission, input_mask_by_auxemission))
    print("input_mask_by_emission", input_mask_by_emission, input_mask_by_emission.shape)
    print("input_mask_by_auxemission", input_mask_by_auxemission, input_mask_by_auxemission.shape)
    print("input_mask_by_allemission", input_mask_by_allemission, input_mask_by_allemission.shape)

    learned_params = model_ckp['learned_params']

    if 'hmm' in model_prefix or 'HMM' in model_prefix:
        if 'ghmm' in model_prefix:
            return
        reg_weights = learned_params.emissions.weights
    elif model_prefix == 'lr':
        reg_weights = learned_params['w']
    else:
        raise Exception(f'wrong prefix={model_prefix}')

    logreg_params = auxem_model_ckp['logreg_params']
    aux_weights = logreg_params['w']
    all_weights = np.concatenate((reg_weights, aux_weights), axis=1)

    print(reg_weights.shape)
    print(aux_weights.shape)
    print(all_weights.shape)

    emission_labels = data_config['emission_labels']
    auxiliary_emission_labels = data_config['auxiliary_emission_labels']

    all_emission_labels = OrderedDict()
    all_emission_labels.update(emission_labels)
    all_emission_labels.update(auxiliary_emission_labels)
    print(emission_labels)
    print(auxiliary_emission_labels)
    print(all_emission_labels)

    input_labels_list = data_config['input_labels_list']
    auxiliary_input_labels_list = data_config['auxiliary_input_labels_list']
    # all_input_labels_list = data_config['input_labels_list'] + data_config['auxiliary_input_labels_list']

    print(data_config['input_labels_list'])
    print(data_config['auxiliary_input_labels_list'])
    # print(all_input_labels_list)

    print(data_config['input_labels'])

    only_plot_inputs = ['fFV', 'mfDist'] if animal == 'male' else ['mFV', 'pulse_i', 'sine_i', 'tap2', 'mfDist']

    for skip_states in [[0], []]:
        if num_states <= 1 and skip_states:
            continue    # skip skip_states if there's only state
        plots.plot_filters_statewise(all_weights, data_config, input_labels_list, auxiliary_input_labels_list, data_config['input_labels'], all_emission_labels, auxiliary_emission_labels, prefix='allemissions',
                                     only_plot_inputs=only_plot_inputs, skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filters(all_weights, data_config, all_emission_labels, filesuffix='allemissions', skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filters(all_weights, data_config, all_emission_labels, filesuffix='allemissions', skip_states=skip_states, sharey='row', savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filters_separate_emissions(all_weights, data_config, all_emission_labels, input_labels, input_mask_by_allemission, filesuffix='allemissions', sharey='row', skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filters_separate_emissions(all_weights, data_config, all_emission_labels, input_labels, input_mask_by_allemission, filesuffix='allemissions', sharey='row', skip_states=skip_states, saveindividual=True, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filter_amplitudes(all_weights, data_config, data_config['input_labels'], all_emission_labels, input_mask_by_allemission, prefix='allemissions', skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filter_amplitudes(all_weights, data_config, data_config['input_labels'], all_emission_labels, input_mask_by_allemission, prefix='allemissions', plot_top_k=5, skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
    return


def generate_together_figures_filters_given(model_dir, all_weights, savefig=True, display=False):
    """Plots, such as filters, that need to be together for emissions and aux-emissions. """

    model_ckp, data_config, _ = load_specific_path(model_dir)
    auxem_model_ckp = load_specific_path_auxem(model_dir)
    if (model_ckp is None) or (auxem_model_ckp is None):
        return

    fig_dir = os.path.join(model_dir, 'figures')
    together_filters_fig_dir = os.path.join(fig_dir, 'together_filters_avgd_fig_dir')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(together_filters_fig_dir, exist_ok=True)

    update_labels(data_config)
    input_labels = data_config['input_labels']
    # model_prefix = model_ckp['prefix']
    num_states = model_ckp['num_states']
    animal = data_config['animal']

    # plot filters for regular+auxem emissions
    input_mask_by_emission = data_config['input_mask_by_emission']
    input_mask_by_auxemission = np.array(auxem_model_ckp['input_mask_by_auxemission'])
    input_mask_by_allemission = np.concatenate((input_mask_by_emission, input_mask_by_auxemission))
    print("input_mask_by_emission", input_mask_by_emission, input_mask_by_emission.shape)
    print("input_mask_by_auxemission", input_mask_by_auxemission, input_mask_by_auxemission.shape)
    print("input_mask_by_allemission", input_mask_by_allemission, input_mask_by_allemission.shape)

    emission_labels = data_config['emission_labels']
    auxiliary_emission_labels = data_config['auxiliary_emission_labels']

    all_emission_labels = OrderedDict()
    all_emission_labels.update(emission_labels)
    all_emission_labels.update(auxiliary_emission_labels)
    print(emission_labels)
    print(auxiliary_emission_labels)
    print(all_emission_labels)

    input_labels_list = data_config['input_labels_list']
    auxiliary_input_labels_list = data_config['auxiliary_input_labels_list']
    # all_input_labels_list = data_config['input_labels_list'] + data_config['auxiliary_input_labels_list']

    print(data_config['input_labels_list'])
    print(data_config['auxiliary_input_labels_list'])
    # print(all_input_labels_list)

    print(data_config['input_labels'])

    only_plot_inputs = ['fFV', 'mfDist'] if animal == 'male' else ['mFV', 'pulse_i', 'sine_i', 'tap2', 'mfDist']

    for skip_states in [[0], []]:
        if num_states <= 1 and skip_states:
            continue    # skip skip_states if there's only state
        plots.plot_filters(all_weights, data_config, all_emission_labels, filesuffix='allemissions', skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filters(all_weights, data_config, all_emission_labels, filesuffix='allemissions', skip_states=skip_states, sharey='row', savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filters_separate_emissions(all_weights, data_config, all_emission_labels, input_labels, input_mask_by_allemission, filesuffix='allemissions', sharey='row', skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filters_separate_emissions(all_weights, data_config, all_emission_labels, input_labels, input_mask_by_allemission, filesuffix='allemissions', sharey='row', skip_states=skip_states, saveindividual=True, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filters_statewise(all_weights, data_config, input_labels_list, auxiliary_input_labels_list, data_config['input_labels'], all_emission_labels, auxiliary_emission_labels, prefix='allemissions',
                                     only_plot_inputs=only_plot_inputs, skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filter_amplitudes(all_weights, data_config, data_config['input_labels'], all_emission_labels, input_mask_by_allemission, prefix='allemissions', skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
        plots.plot_filter_amplitudes(all_weights, data_config, data_config['input_labels'], all_emission_labels, input_mask_by_allemission, prefix='allemissions', plot_top_k=5, skip_states=skip_states, savefig=savefig, fig_dir=together_filters_fig_dir, display=display)
    return


def generate_figures_filters_given_2datasets(data_config1, data_config2, all_weights1, all_weights2, fig_dir=None, savefig=True, display=False):

    update_labels(data_config1)
    update_labels(data_config2)
    input_labels = data_config1['input_labels']
    num_states = all_weights1.shape[0]
    assert num_states == all_weights2.shape[0]

    # plot filters for regular emissions
    input_mask_by_emission = data_config1['input_mask_by_emission']

    emission_labels = data_config1['emission_labels']

    for skip_states in [[0, 1, 2, 3]]:
        plots.plot_filters_separate_emissions_2datasets(all_weights1, all_weights2, data_config1, data_config2, emission_labels, input_labels, input_mask_by_emission, filesuffix='cmp2datasets', sharey=None, skip_states=skip_states, saveindividual=True, savefig=savefig, fig_dir=fig_dir, display=display)
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
        if len(xlim_) != 2:
            continue
        xlim = (int(xlim_[0]), int(xlim_[1]))
        len_traj = xlim[1] - xlim[0]
        xlim_orig = (int(model_ckp[data_key][dwnsmpl_key][batch][xlim[0]]), int(model_ckp[data_key][dwnsmpl_key][batch][xlim[1]]))
        plots.plot_state_probs(model_ckp[data_key][f'{prefix}_state_probs'], model_config, data_config, batch, effective_fps, xlim=xlim, xlim_orig=xlim_orig, prefix=prefix, suffix=suffix, savefig=savefig, fig_path=f'{probs_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)
        plots.plot_trajectories(model_ckp, model_config, data_config, batch, states_in_bgr=False, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)
        plots.plot_trajectories_statewise(model_ckp, model_config, data_config, batch, states_in_bgr=False, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}_perstate{suffix}', display=display)

        # plots.plot_comparison_probs(model_ckp[data_key][f'{prefix}_state_probs'], model_ckp[data_key][f'{prefix_data}_fwd_state_probs'], model_config, batch, effective_fps, xlim=xlim, xlim_orig=xlim_orig, prefix_data=prefix_data, suffix=suffix, savefig=savefig, fig_path=f'{probs_dir}/{prefix}{batch}_xlim={xlim}{suffix}_.pdf', display=display)
        # plots.plot_trajectories(model_ckp, model_config, data_config, batch, states_in_bgr=True, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)
        # plots.plot_trajectories2D(model_ckp, model_config, data_config, batch, states_in_bgr=True, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs2d_dir}/{prefix}{batch}_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)
        # plots.plot_trajectories_w_partner(model_ckp, model_config, data_config, batch, prefix=prefix, suffix=suffix, xlim=xlim, xlim_orig=xlim_orig, savefig=savefig, fig_path=f'{trajs_dir}/{prefix}{batch}_w_partner_{len_traj}_{i}_xlim={xlim}{suffix}.pdf', display=display)

        if gen_corr_video:
            clip_session(os.path.join('/Volumes/murthy/usingla/gold_dataset/wt/mp4', key_b.replace(".h5", ".mp4")), xlim_orig, output_path=f'{trajs_dir}/{prefix}{batch}_xlim_orig={xlim_orig}_xlim={xlim}{suffix}.mp4')
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

        for batch in np.random.choice(n_sessions, size=min(10, n_sessions)):
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
    """
    2d trajectories from data. 9 clips in a grid.
    """

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
        smoothed_track_data = joblib.load('../data/wt/smoothed_track_data_81_jan1.pkl')
    elif dataset == 'wt_fred':
        smoothed_track_data = joblib.load('../data/wt_fredcleaned/smoothed_track_data_11_jan1.pkl')

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

                fTrx = smoothed_track_data[key_b]['fTrx']
                mTrx = smoothed_track_data[key_b]['mTrx']

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
    # trajs2d_dir = os.path.join(fig_dir, 'trajs2d')
    probs_dir = os.path.join(fig_dir, 'probs')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(trajs_dir, exist_ok=True)
    # os.makedirs(trajs2d_dir, exist_ok=True)
    os.makedirs(probs_dir, exist_ok=True)

    def f(prefix):

        data_key = f'{prefix}_data'
        n_sessions = len(model_ckp[data_key][f'{prefix}_session_keys'])

        # for batch in np.random.choice(n_sessions, size=min(5, n_sessions)):
        # for batch in [35, 5]:
        for batch in range(n_sessions):
            # batch = 24 if prefix == 'train' else batch
            # batch = 10 if prefix == 'test' else batch
            key_b = model_ckp[data_key][f'{prefix}_session_keys'][batch]
            num_timestamps = model_ckp[data_key][f'{prefix}_stateseq'][batch].shape[0]
            print("batch", batch, "key_b", key_b, "num_timestamps", num_timestamps)
            windows = get_windows_to_plot(effective_fps, num_timestamps)
            # print("windows", windows)
            plot_xlims(model_dir, windows, batch, prefix, trajs_dir, None, probs_dir, savefig=savefig, display=display, gen_corr_video=gen_corr_video)
            lastwindows = get_cop_window_to_plot(effective_fps, num_timestamps)
            plot_xlims(model_dir, lastwindows, batch, prefix, trajs_dir, None, probs_dir, suffix='(final 30 seconds)', savefig=savefig, display=display, gen_corr_video=gen_corr_video)
            fullwindows = get_full_window_to_plot(effective_fps, num_timestamps)
            plot_xlims(model_dir, fullwindows, batch, prefix, trajs_dir, None, probs_dir, suffix='(whole session)', savefig=savefig, display=display, gen_corr_video=gen_corr_video)
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

