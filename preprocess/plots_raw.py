"""
Script to plot raw sensory input or behavioral output traces for Figure 1 (behavior).
Not used otherwise.
"""

import os

import joblib
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from leaprig import WT_DATA
from preprocess.get_designmatrix import create_x_and_y_windows
from preprocess.colors import *


def get_feat(sessions_features, s, f_name):
    sf = sessions_features[s]
    if f_name in ['mFV', 'mLS', 'mFA', 'mLA', 'mLV', 'mfDist', 'fDistWall', 'fFV', 'fLS', 'fLV', 'fFV', 'fFS', 'fLS', 'fLV', 'fFA', 'mFV', 'mFS', 'mLS', 'mLV']:
        ts = sf[f_name]
    elif f_name in ['song', 'sine_i', 'pulse_i', 'song_i', 'tap2']:
        ts = sf[f_name]
    elif f_name in ['fmAng']:
        ts = np.radians(sf['fmAng'])
    elif f_name in ['fmAng_cos']:
        ts = np.radians(np.abs(sf['fmAng']))
        ts = np.cos(ts)    # cos: front to back
    elif f_name in ['fmAng_sin']:
        ts = np.radians(sf['fmAng'])
        ts = np.sin(ts)    # sin: left or right of the fly
    elif f_name in ['wingAlign']:
        ts = np.min([sf['wingLAristaLAlignAng'],
                       sf['wingRAristaRAlignAng'],
                       sf['wingLAristaRAlignAng'],
                       sf['wingRAristaLAlignAng']], axis=0)
    elif f_name in ['song_directedlr', 'sine_i_directedlr', 'pulse_i_directedlr2', 'song_i_directedlr2', 'tap2_directedlr']:
        f_name_ = f_name.split('_directed')[0]
        ts = sf[f_name_] * np.sign(np.sin(np.radians(sf['fmAng'])))
    elif f_name in ['fAV']:
        ts = sf['fTheta']
        dfTheta = np.diff(ts, prepend=ts[0])
        ts = np.where(np.abs(dfTheta) > 90, 0, dfTheta)
    elif f_name in ['mAV']:
        ts = sf['mTheta']
        dmTheta = np.diff(ts, prepend=ts[0])
        ts = np.where(np.abs(dmTheta) > 90, 0, dmTheta)
    elif f_name in ['wingFlickTheta']:
        ts = sf.get('wingFlickAngle', sf.get('wingMaxAngle'))
        ts = ts * sf['wingFlick']
    elif f_name in ['wingFlickBin']:
        ts = sf['wingFlick']
    else:
        raise Exception(f'unsupported {f_name} input feature.')
    return ts


def get_x_and_y_data(datacls, sessions_features, config):

    x_labels = config['input_labels']
    y_labels = config['emission_labels']
    input_raw_each_dim = config['input_raw_each_dim']
    predict_window_size = config['predict_window_size']
    input_raw_overlap = config['input_raw_overlap']
    predict_gap_size = config['predict_gap_size']

    inputs_raw = []
    outputs_raw = []
    session_keys = []

    for s_i, s in enumerate(sessions_features):

        if s_i == 0:
            print('Available features:', sessions_features[s].keys())
        session_len = len(sessions_features[s]['mFV'])
        print(f"Session {s_i} ({s}) length: {session_len}.")

        num_timesteps = session_len

        print(f"Proceeding with the session {s_i}.")
        session_keys.append(s)
        session_copulation = datacls.get_copulation_bool_from_session(s, session_len)
        print(f'Session {s_i} copulation={session_copulation} session_len={session_len}.')

        input_windows, _ = create_x_and_y_windows(num_timesteps, x_size=input_raw_each_dim,
                                                               y_size=predict_window_size, x_overlap=input_raw_overlap,
                                                               y_gap_size=predict_gap_size)

        s_start_frame = (session_len-num_timesteps)     # index of the first frame of this session used for modeling
        s_windows = input_windows + s_start_frame

        # INPUTS
        i_feats = []
        for _ in x_labels:
            f = get_feat(sessions_features, s, _)[s_windows]
            i_feats.append(f)
        print(f"session {s_i} inp computed")
        s_inputs = np.hstack(i_feats)
        print(f"session {s_i} inp processed")

        # OUTPUTS
        o_feats = []
        for _ in y_labels:
            f = get_feat(sessions_features, s, _)[s_windows]
            o_feats.append(f)
        s_outputs = np.hstack(o_feats)
        print(f"session {s_i} output processed")

        inputs_raw.append(s_inputs)
        outputs_raw.append(s_outputs)
        if s_i == 5:
            break

    inputs_raw = np.array(inputs_raw, dtype=object)
    outputs_raw = np.array(outputs_raw, dtype=object)
    return inputs_raw, outputs_raw


def plot_inputs(batch, idxs, inputs_raw, data_config, small=False, savedir=None, display=True):
    input_labels = data_config['input_labels']
    n_feats = len(input_labels)
    feat_size = data_config['input_raw_each_dim']
    i_features = inputs_raw[batch].reshape(-1, n_feats, feat_size)

    fontsize = 14 if small else 18
    figwidth = 2 if small else 5
    figheight = n_feats * 0.5

    for idx in idxs:
        print("batch", batch, "idx", idx)
        fig, axes = plt.subplots(n_feats, 1, figsize=(figwidth, figheight), sharex=True)
        i = 0
        for _ in input_labels:
            r = np.r_[feat_size // 2:] if small else np.r_[0:feat_size]
            axes[i].plot(i_features[idx][i][r], color=data_config['input_label_colors'][_], linewidth=2)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            # axes[i].text(-0.5, 0, input_labels[_], ha='right')
            axes[i].text(-0.02, 0.5, input_labels[_], transform=axes[i].transAxes, fontsize=fontsize,
                         ha='right', va='center', color=data_config['input_label_colors'][_])
            i += 1
        fig.align_ylabels()
        plt.tight_layout()

        if savedir: plt.savefig(os.path.join(savedir, f'i_small={small}_batch={batch}_ix={idx}.pdf'), transparent=True, dpi=300, bbox_inches='tight')
        if display: plt.show()
        plt.close()
    return


def plot_emissions(batch, idxs, emissions, data_config, savedir=None, display=True):
    emission_labels = data_config['emission_labels']
    n_feats = len(emission_labels)
    feat_size = data_config['input_raw_each_dim']
    i_features = emissions[batch].reshape(-1, n_feats, feat_size)

    for idx in idxs:
        print("batch", batch, "idx", idx)
        fig, axes = plt.subplots(n_feats, 1, figsize=(6, n_feats * 0.75), sharex=True)
        i = 0
        for _ in emission_labels:
            axes[i].plot(i_features[idx][i], color=data_config['emission_label_colors'][_], linewidth=2)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].text(-0.02, 0.5, emission_labels[_], transform=axes[i].transAxes, fontsize=18,
                         ha='right', va='center', color=data_config['emission_label_colors'][_])
            i += 1
        fig.align_ylabels()
        plt.tight_layout()
        if savedir: plt.savefig(os.path.join(savedir, f'e_batch={batch}_ix={idx}.pdf'), transparent=True, dpi=300, bbox_inches='tight')
        if display: plt.show()
        plt.close()
    return


def extract_female(source):

    data_config = {}

    sessions_features = joblib.load('../data/wt/sessions_features_74_sep5.pkl')
    datacls = WT_DATA

    fps = sessions_features.get('fps', datacls.fps)
    data_config['source'] = source
    data_config['orig_fps'] = fps
    data_config['input_raw_each_dim'] = 3*fps
    data_config['predict_gap_size'] = 0     # any gap between x inputs and y output
    data_config['input_raw_overlap'] = fps//30    # move input window forward by 33ms (TODO keep this same as predict window size?)
    data_config["predict_window_size"] = fps//30  # averaging emission over this window size
    data_config['effective_fps'] = data_config['orig_fps'] / data_config["predict_window_size"]
    data_config['basis_transformed'] = 'cos'  # 'cos', 'smooth', or 'identity'
    data_config['ncos'] = 4
    data_config['input_labels'] = OrderedDict({
        'mFV': 'mFV',
        'mLS': 'mLS',
        'mfDist': 'mfDist',

        # 'fmAng': "fmAng",
        'fmAng_cos': r"$\it{cos}$(fmAng)",
        'fmAng_sin': r"$\it{sin}$(fmAng)",

        'wingAlign': 'wingArisAng',
        # 'pfast_i': 'pulse song',
        'pulse_i': 'pulse song',
        # 'pfast_i_directedlr': 'pulse song\nx side',
        'sine_i': 'sine song',
        # 'sine_i_directedlr': 'sine song\nx side',

        'tap2': 'tap',
        # 'tap2_directedlr': 'tap\nx side',

        # 'fDistWall': 'distWall',
    })
    data_config['input_label_colors'] = OrderedDict({
        'mFV': IC,
        'mLS': IC,
        'mfDist': 'k',

        'fmAng': 'k',
        'fmAng_cos': input_label_colors['mfDist'],
        'fmAng_sin': input_label_colors['mfDist'],

        'wingAlign': IC,
        # 'pfast_i': input_label_colors['pfast_i'],
        'pulse_i': input_label_colors['pfast_i'],
        'sine_i': input_label_colors['sine_i'],
        # 'pfast_i_directedlr': input_label_colors['pfast_i_directedlr'],
        'pulse_i_directedlr2': input_label_colors['pulse_i_directedlr2'],
        'sine_i_directedlr2': input_label_colors['sine_i_directedlr2'],

        'tap2': input_label_colors['tap2'],
        'tap2_directedlr2': input_label_colors['tap2_directedlr2'],

        # 'fDistWall': 'distWall',
    })
    data_config['emission_labels'] = OrderedDict({
        'fFV': 'forward velocity\n(fFV)',
        'fLV': 'lateral velocity\n(fLV)',
        'fAV': 'angular velocity\n(fAV)',
        'wingFlickBin': 'wing flick',
    })
    data_config['emission_label_colors'] = OrderedDict({
        'fFV': EC,
        'fLV': EC,
        'fAV': EC,
        'wingFlickBin': EC,
    })

    inputs_raw, emissions = get_x_and_y_data(datacls, sessions_features, data_config)

    print("====", inputs_raw[0].shape)

    introfigs_dir = '../../paper figs/figure_behavior/datasamples'
    os.makedirs(introfigs_dir, exist_ok=True)

    # Plot each feature
    batch = 2
    np.random.seed()
    i_features = inputs_raw[batch].reshape(-1, len(data_config['input_labels']), data_config['input_raw_each_dim'])
    idxs = np.random.choice(i_features.shape[0], size=100)
    for ix in idxs:
        ix = 15624  # 190724_112816_wt_16276625_rig2.1.h5
        print("batch", batch, "ix", ix)
        plot_inputs(batch, [ix], inputs_raw, data_config, small=False, savedir=introfigs_dir, display=False)
        plot_inputs(batch, [ix], inputs_raw, data_config, small=True, savedir=introfigs_dir, display=False)
        plot_emissions(batch, [ix], emissions, data_config, savedir=introfigs_dir, display=False)
        break

    return


if __name__ == '__main__':
    src = 'wt'
    extract_female(src)
