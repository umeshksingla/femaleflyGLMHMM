import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.ticker import FixedLocator
from matplotlib.patches import FancyArrowPatch, Rectangle
from functools import reduce
import seaborn as sns
import os
import networkx as nx
import pandas as pd

import numpy as np
from dynamax.utils.plotting import CMAP, COLORS
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import f1_score
from preprocess.colors import *
from utilities.io import get_feat_windows, get_event_onsets, basis_invtransform_one_by_one, load_specific_path

# -- Fonts --
mpl.rcParams['font.size'] = 18  # Panel label
# mpl.rcParams['font.family'] = 'Arial'
# mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'

# -- Axes --
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.grid.axis'] = 'y'
mpl.rcParams['grid.color'] = 'black'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.ymargin'] = 0
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["legend.fontsize"] = 20
plt.rcParams['axes.titlesize'] = 20

# -- Ticks and tick labels --
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

# -- Figure size --
# plt.rcParams['figure.figsize'] = (6, 4)
# plt.rcParams['figure.dpi'] = 300
# mpl.rcParams['legend.frameon'] = False

# -- Saving Options --
# plt.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# rcParams['savefig.transparent'] = True

# -- Plot Styles --
mpl.rcParams['lines.linewidth'] = 1
#################


def plot_legends(num_states, data_config, savefig=False, fig_dir=None, display=True):

    for z in range(5):
        fig, ax = plt.subplots(figsize=(2.4, 1.2))
        ax.plot([0, 1], [1, 1], c=COLORS[z], linewidth=6)
        ax.text(1.2, 1, f'Dataset 1', va='center', fontsize='small')
        ax.plot([0, 1], [0, 0], c=COLORS[z], linewidth=6, alpha=0.5)
        ax.text(1.2, 0, f'Dataset 2', va='center', fontsize='small')
        # ax.set_ylim([0.5, -num_states-0.5])
        ax.axis('off')
        plt.tight_layout()
        if savefig:
            plt.savefig(f'{fig_dir}/legend_z={z}.pdf', dpi=300, transparent=True, bbox_inches='tight')
        if display:
            plt.show()
        plt.close()

    fig, ax = plt.subplots(figsize=(2.4, 1.7))
    ax.plot([0, 1], [1, 1], c=COLORS[0], linewidth=6)
    ax.text(1.2, 1, f'LEAP\ndataset', va='center', fontsize='small')
    ax.plot([0, 1], [0, 0], c=COLORS[0], linewidth=6, linestyle='--')
    ax.text(1.2, 0, f'16mic\ndataset', va='center', fontsize='small')
    # ax.set_ylim([0.5, -num_states-0.5])
    ax.axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{fig_dir}/legend_datasets.pdf', dpi=300, transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(2.25, 2.25))

    for i, z in enumerate(range(num_states-1, -1, -1)):
        ax.plot([0, 1], [-i, -i], c=COLORS[z], linewidth=4)
        ax.text(1.2, -i, f'State {z+1}', va='center', fontsize='medium')

    ax.set_ylim([0.5, -num_states])
    ax.axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{fig_dir}/legend_z.pdf', dpi=300, transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(2.25, 1.25))
    ax.plot([0, 1], [0, 0], c=EC, linewidth=6)
    ax.text(1.2, 0, f'female', va='center', fontsize='medium')
    ax.plot([0, 1], [1, 1], c=IC, linewidth=6)
    ax.text(1.2, 1, f'male', va='center', fontsize='medium')
    # ax.set_ylim([0.5, -num_states-0.5])
    ax.axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{fig_dir}/legend_fly.pdf', dpi=300, transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(2.75, 1.25))
    ax.plot([0, 1], [1, 1], c='gray', linewidth=6)
    ax.text(1.2, 1, f'Data', va='center', fontsize='medium')
    ax.plot([0, 1], [0, 0], c=EC, linewidth=6)
    ax.text(1.2, 0, f'GLM-HMM', va='center', fontsize='medium')
    # ax.set_ylim([0.5, -num_states-0.5])
    ax.axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{fig_dir}/legend_prediction.pdf', dpi=300, transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(2.25, 2))
    feats = ['mFV', 'pulse_i_directedlr', 'sine_i_directedlr', 'tap2_directedlr']
    i_labels = data_config['input_labels_jr']
    for i, f in enumerate(feats):
        if f in i_labels:
            ax.plot([0, 1], [-i, -i], c=input_label_colors[f], linewidth=4)
            ax.text(1.2, -i, i_labels[f], va='center', fontsize='small')

    ax.set_ylim([-len(feats), 0.5])
    ax.axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{fig_dir}/legend_inputs.pdf', dpi=300, transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(2.25, 2))
    feats = ['mFV', 'pfast_i', 'sine_i', 'tap2']
    i_labels = data_config['input_labels_jr']
    for i, f in enumerate(feats):
        if f in i_labels:
            ax.plot([0, 1], [-i, -i], c=input_label_colors[f], linewidth=4)
            ax.text(1.2, -i, i_labels[f], va='center', fontsize='small')

    ax.set_ylim([-len(feats), 0.5])
    ax.axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{fig_dir}/legend_inputs0.pdf', dpi=300, transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()
    return


def plot_hmm_data_whole_session_with_states_on_top(predicted_emissions, true_emissions, predicted_states, config, model_label=None, xlim=None, xlim_orig=None, y_labels=None, title=None, savefig=False, fig_path=None, display=True):

    emission_dim = predicted_emissions.shape[-1]

    lent = xlim[1] - xlim[0]
    if lent > 8000:
        fig_width = 24
    elif lent < 600:
        fig_width = 8
    elif 600 < lent and lent < 8000:
        fig_width = 15

    fig, axs = plt.subplots(emission_dim+1, 1, figsize=(fig_width, 2.1*emission_dim+0.1), sharex=True, gridspec_kw={'height_ratios': [0.25] + [1]*emission_dim })

    xlim_ = np.r_[xlim[0]:xlim[1] + 1].astype(int)

    ax = axs[0]
    ax.imshow(predicted_states[xlim_][None, :], aspect="auto", interpolation="none", cmap=CMAP, vmin=0,
              vmax=len(COLORS) - 1,
              extent=(xlim_[0], xlim_[-1], 0, 0.1), alpha=0.7)

    d = 0
    for _ in y_labels:
        ax = axs[d+1]
        ax.plot(xlim_, true_emissions[xlim_, d], '-', color="gray", label='Data')
        ax.plot(xlim_, predicted_emissions[xlim_, d], '-', color=EC, linewidth=2, label=f'{model_label}')
        # ax.set_ylabel(y_labels[_], c=EC)
        # ax.margins(y=0.05)
        d += 1

    for ax in axs:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

    # plt.xlabel("Time (s)")
    # ax = axs[1]
    # ax.legend(loc='upper right')
    # plt.suptitle(title)
    fig.align_ylabels()
    plt.tight_layout()

    if savefig: plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_hmm_data_whole_session_perstate_states_on_top(predicted_emissions, predicted_emissions_per_state, true_emissions, predicted_states, config, model_config, o_index=0, model_label=None, xlim=None, xlim_orig=None, y_labels=None, title=None, savefig=False, fig_path=None, display=True):

    num_states = model_config['num_states']

    fig, axs = plt.subplots(num_states+2, 1, figsize=(12, 1.5*num_states+0.1), sharex=True,
                            # sharey=True,
                            gridspec_kw={'height_ratios': [0.25] + [1]*(num_states+1)}
                            )

    xlim_ = np.r_[xlim[0]:xlim[1] + 1].astype(int)

    ax = axs[0]
    ax.imshow(predicted_states[xlim_][None, :], aspect="auto", interpolation="none", cmap=CMAP, vmin=0,
              vmax=len(COLORS) - 1,
              extent=(xlim_[0], xlim_[-1], 0, 0.1), alpha=0.7)

    ax = axs[1]
    ax.plot(xlim_, true_emissions[xlim_, o_index], '-', color="gray", label='Data')   # plotting only fFV
    ax.plot(xlim_, predicted_emissions[xlim_, o_index], '-', color=EC, linewidth=2, label=f'{model_label}')
    ax.set_ylabel(f'{num_states}-state\n{model_label}', rotation=360, ha='left', va='center')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    for z in range(num_states):
        ax = axs[z+2]
        ax.plot(xlim_, true_emissions[xlim_, o_index], '-', color="gray", label='Data')
        ax.plot(xlim_, predicted_emissions_per_state[xlim_, z, o_index], '-', color=COLORS[z], linewidth=2, label=f'{model_label}')
        ax.set_ylabel(f'State {z+1}', color=COLORS[z], rotation=360, ha='left', va='center')
        ax.sharey(axs[1])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    for ax in axs:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

    # plt.xlabel("Time (s)")
    # ax = axs[1]
    # ax.legend(loc='upper right')
    # plt.suptitle(title)
    # fig.align_ylabels()
    fig.supylabel(list(y_labels.values())[o_index].replace('\n', ' '), ha='center', va='center')
    plt.tight_layout()

    if savefig: plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_hmm_data_whole_session_with_aux_with_states(predicted_emissions, true_emissions, aux_data, predicted_states, config, model_label=None, xlim=None, xlim_orig=None,  y_labels=None, aux_labels=None, title=None, savefig=False, fig_path=None, display=True):
    """Plot emissions and aux_data vs. time"""
    emission_dim = predicted_emissions.shape[-1]
    aux_dim = aux_data.shape[-1]
    fig, axs = plt.subplots(emission_dim+aux_dim, 1, figsize=(15, 15), sharex=True)

    xlim_ = np.r_[xlim[0]:xlim[1]+1].astype(int)

    d = 0
    for _ in y_labels:
        ax = axs[d]

        max_value = max(abs(predicted_emissions[xlim_, d]).max(), abs(true_emissions[xlim_, d]).max())
        lim = 1.05 * max_value
        ax.imshow(predicted_states[xlim_][None, :], aspect="auto", interpolation="none", cmap=CMAP, vmin=0, vmax=len(COLORS)-1,
                  extent=(xlim_[0], xlim_[-1], -lim, lim), alpha=0.7)
        ax.plot(xlim_, true_emissions[xlim_, d], "k-", alpha=0.6, label='Data')
        ax.plot(xlim_, predicted_emissions[xlim_, d], 'm-', linewidth=2, label=f'{model_label}')
        ax.set_ylabel(y_labels[_], fontsize='xx-small', c=EC)
        ax.set_ylim([-lim, lim])
        ax.margins(y=0.05)

        xt = np.linspace(xlim_[0], xlim_[-1], num=5)
        pws = config['predict_window_size']
        init_period = config['input_raw_each_dim']
        orig_fps = config['orig_fps']
        ax.set_xticks(xt)
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.set_xticklabels([f"{round((pws * x + init_period)/orig_fps, 1)}" for x in xt])

        d += 1
        axs[0].legend(loc='upper right')

    a = 0
    for _ in aux_labels:
        ax = axs[d]

        max_value = abs(aux_data[xlim_, a]).max()
        if max_value == 0.:
            max_value = 1
        lim = 1.05 * max_value
        ax.imshow(predicted_states[xlim_][None, :], aspect="auto", interpolation="none", cmap=CMAP, vmin=0, vmax=len(COLORS)-1,
                  extent=(xlim_[0], xlim_[-1], -lim, lim), alpha=0.7)
        ax.plot(xlim_, aux_data[xlim_, a], "g-", alpha=0.8, label='Partner Data')
        ax.set_ylabel(aux_labels[_], fontsize='xx-small', c='g')
        ax.set_ylim([-lim, lim])
        ax.margins(y=0.05)
        # ax.set_yticks([0])

        xt = np.linspace(xlim_[0], xlim_[-1], num=5)
        pws = config['predict_window_size']
        init_period = config['input_raw_each_dim']
        orig_fps = config['orig_fps']
        ax.set_xticks(xt)
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.set_xticklabels([f"{round((pws * x + init_period)/orig_fps, 1)}" for x in xt])

        if a == 0:
            axs[d].legend(loc='upper right')
        d += 1
        a += 1

    plt.xlabel("Time (s)")
    plt.suptitle(title)
    fig.align_ylabels()
    plt.tight_layout()
    if savefig: plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_filters_statewise(orig_weights, data_config, input_labels_list, aux_input_labels_list, input_labels, y_labels, y_aux_labels, prefix, sharey='row', axtitles=True, skip_states=[], only_plot_inputs=None, savefig=False, fig_dir=None, display=True):
    # print(weights.shape)

    num_states = orig_weights.shape[0]
    emission_dim = orig_weights.shape[1]
    # filter_len = orig_weights.shape[-1]
    # input_labels = data_config['input_labels']
    # y_labels = data_config['emission_labels']
    directional_y_variables = data_config['directional_variables']
    n_inputs = len(data_config['input_labels_list'])
    basis = data_config['basis']

    if not only_plot_inputs:
        only_plot_inputs = input_labels_list

    print(orig_weights.shape)
    weights = basis_invtransform_one_by_one(orig_weights, basis, n_inputs)  # shape
    print(weights.shape)

    eff_num_states = num_states - len(skip_states)
    fig, axs = plt.subplots(emission_dim, eff_num_states, figsize=(2.5 * eff_num_states + 3, 2.4 * len(y_labels)), sharey=sharey)

    n_feats = 0
    for d, _ in enumerate(y_labels):
        # print(d, _)
        w = weights[:, d]
        # w = w / np.linalg.norm(w)   # normalize across states
        if _ in y_aux_labels:
            n_feats = 0     # reset
        base_idx = n_feats
        s = 0
        for z in range(num_states):
            print(z)
            if z in skip_states:
                continue
            # if z == 0 and num_states > 1: continue

            if emission_dim > 1 and eff_num_states > 1:
                ax = axs[d, s]
            elif emission_dim > 1 and eff_num_states == 1:
                ax = axs[d]
            elif emission_dim == 1 and eff_num_states > 1:
                ax = axs[s]
            else:
                ax = axs

            if ax.get_subplotspec().is_first_row() and prefix != 'aux_emissions':
                ax.set_title(f'State {z + 1}', color=COLORS[z])
            # if ax.get_subplotspec().is_first_col():
            #     ax.set_ylabel(f'{y_labels[_]}', color=EC)

            for __ in only_plot_inputs:
                if _ in directional_y_variables:
                    __ = __ + '_directedlr2'

                if _ in y_aux_labels:
                    stim = base_idx + aux_input_labels_list[base_idx:].index(__)
                else:
                    stim = base_idx + input_labels_list[base_idx:].index(__)       # need base_idx to skip to where stim for this y start (since we have overlapping stim names for different y's)

                # print(__, "stim idx", stim, "base_idx", base_idx)
                ax.plot(np.arange(-w[z, stim].shape[-1], 0)/data_config['orig_fps'], w[z, stim], linewidth=3, label=input_labels[__], color=input_label_colors[__])
                ax.axhline(0, ls=':', c='k', lw=0.5)
                ax.set_xticks([-w[0, stim].shape[-1]//data_config['orig_fps'], 0])
                ax.margins(y=0.05)
                if ax.get_subplotspec().is_last_row():
                    ax.set_xlabel('Time (s)')

            if ax.get_subplotspec().is_last_col():
                ax.legend(loc='upper right', bbox_to_anchor=(3, 1), borderaxespad=0.)
            s += 1
        n_feats += len(y_labels[_])

    if emission_dim > 1 and eff_num_states > 1:
        fig.align_ylabels(axs[:, 0])
    elif emission_dim > 1 and eff_num_states == 1:
        fig.align_ylabels(axs[:])
    # fig.supylabel('Filter amplitude (a.u.)')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{prefix}_filters_statewise_skip_states={skip_states}_new.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_filters(orig_weights, data_config, y_labels, filesuffix, skip_states=[], sharey=False, savefig=False, fig_dir=None, display=True):

    num_states = orig_weights.shape[0]
    emission_dim = orig_weights.shape[1]
    filter_len = orig_weights.shape[-1]

    input_labels_list = data_config['input_labels_list']
    input_labels = data_config['input_labels']
    n_inputs = len(input_labels_list)
    assert data_config['ncos'] == filter_len // n_inputs
    basis = data_config['basis']
    # print(">>>> obtained weights", orig_weights)

    print("orig_weights.shape", orig_weights.shape) # (num_states, emission_dim, n_inputs * ncos), e.g. (5, 4, 84)
    weights = basis_invtransform_one_by_one(orig_weights, basis, n_inputs)     # shape
    print("weights.shape", weights.shape)   # (num_states, emission_dim, n_inputs, orig filter_len), e.g. (5, 4, 21, 450)

    fig, axs = plt.subplots(emission_dim, n_inputs, figsize=(3 * len(input_labels_list), 2.8 * len(y_labels)), sharey=sharey)
    d = 0
    for _ in y_labels:
        w = weights[:, d]
        stim = 0
        for __ in input_labels_list:

            ax = axs[d, stim] if emission_dim > 1 else axs[stim]

            for z in range(num_states):
                if z in skip_states:
                    continue
                ax.plot(np.arange(-w[z, stim].shape[-1], 0)/data_config['orig_fps'], w[z, stim], color=COLORS[z], linewidth=3, label=f'State {z+1}')

            ax.set_title(input_labels[__])
            # if stim == 0:
            #     ax.set_ylabel(f'{y_labels[_]}\nfilter amplitude (a.u.)', fontsize='x-small', color=EC)

            ax.axhline(0, ls=':', c='k', lw=0.5)
            maxtime = w[0, stim].shape[-1]//data_config['orig_fps']
            ax.set_xticks(np.linspace(-maxtime, 0, num=maxtime+1), labels=[-maxtime] + ['']*(maxtime-1) + [0])
            ax.margins(y=0.05)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel('Time (s)')
            stim += 1
        d += 1

    # ax = axs[0, 0] if emission_dim > 1 else axs[0]
    # ax.legend(loc='upper left')
    # fig.supxlabel("Time relative to prediction (s)")
    # fig.supylabel('Filter amplitude (a.u.)')
    if emission_dim > 1: fig.align_ylabels(axs[:, 0])
    plt.margins(0.02)
    plt.tight_layout()
    fig.canvas.draw()

    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{filesuffix}_filters_skip_states={skip_states}_sharey={sharey}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_filters_separate_emissions(orig_weights, data_config, y_labels, input_labels, input_mask_by_emission, filesuffix, skip_states=[], sharey=False, saveindividual=False, savefig=False, fig_dir=None, display=True):

    num_states = orig_weights.shape[0]

    # input_labels = data_config['input_labels']
    basis = data_config['basis']
    # print(">>>> obtained weights", orig_weights, orig_weights.shape)

    # input_mask_by_emission = data_config['input_mask_by_emission']

    for d, _ in enumerate(y_labels):
        print("y_labels[_]", _, y_labels[_])

        n_inputs_d = len(y_labels[_])
        e_mask = input_mask_by_emission[d]
        print("e_mask", e_mask)
        weights_d = orig_weights[:, [d]][..., e_mask == 1]      # (num states, 1, transformed filterlen), e.g. (5, 1, 28)
        print("weights_d before basis tr", weights_d.shape, "n_inputs_d", n_inputs_d)
        weights_d = basis_invtransform_one_by_one(weights_d, basis, n_inputs=n_inputs_d)[:, 0]
        print("weights_d after basis tr", weights_d.shape)      # (num states, n_inputs_d, orig filterlen), e.g. (5, 7, 450)
        fig, axs = plt.subplots(1, n_inputs_d, figsize=(3 * n_inputs_d, 3), sharey=sharey)

        stim = 0
        for __ in y_labels[_]:
            ax = axs[stim]

            for z in range(num_states):
                if z in skip_states:
                    continue
                ax.plot(np.arange(-weights_d[z, stim].shape[-1], 0)/data_config['orig_fps'], weights_d[z, stim], color=COLORS[z], linewidth=3, label=f'State {z+1}')

            ax.set_title(input_labels[__])

            ax.axhline(0, ls=':', c='k', lw=0.5)
            maxtime = weights_d[0, stim].shape[-1]//data_config['orig_fps']
            n_ticks_inbetween_integers = 4
            ax.set_xticks(np.linspace(-maxtime, 0, num=maxtime*n_ticks_inbetween_integers+1), labels=[-maxtime] + reduce(lambda x, i: x+i, [['']*(n_ticks_inbetween_integers-1) + [int(i)] for i in np.arange(-maxtime+1, 1)]))
            ax.margins(y=0.05)
            if saveindividual:
                ax.yaxis.set_tick_params(labelleft=True)  # ensure ytick labels are shown
            if ax.get_subplotspec().is_last_row() or saveindividual:
                ax.set_xlabel('Time (s)')
            stim += 1

        plt.margins(0.05)
        plt.tight_layout()
        fig.canvas.draw()

        if savefig:
            fig.savefig(os.path.join(fig_dir, f'{filesuffix}_filters_skip_states={skip_states}_sharey={sharey}_separate_emissions_{d}_{_}.pdf'), bbox_inches='tight', dpi=300, transparent=True)

        if saveindividual:
            # Save individual subplots
            os.makedirs(os.path.join(fig_dir, f'{filesuffix}_individual_filters'), exist_ok=True)
            stim = 0
            for __ in y_labels[_]:
                ax = axs[stim]
                extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(os.path.join(fig_dir, f'{filesuffix}_individual_filters', f'skip_states={skip_states}_subplot_{_}_{__}.pdf'), dpi=300, bbox_inches=extent, transparent=True)
                stim += 1

        if display: plt.show()
        plt.close()
    return


def plot_filters_separate_emissions_2datasets(orig_weights1, orig_weights2, data_config1, data_config2, y_labels, input_labels, input_mask_by_emission, filesuffix, skip_states=[], sharey=False, saveindividual=False, savefig=False, fig_dir=None, display=True):

    num_states = orig_weights1.shape[0]

    basis1 = data_config1['basis']
    basis2 = data_config2['basis']

    for d, _ in enumerate(y_labels):
        print("y_labels[_]", _, y_labels[_])

        n_inputs_d = len(y_labels[_])
        e_mask = input_mask_by_emission[d]
        print("e_mask", e_mask)
        weights1_d = basis_invtransform_one_by_one(orig_weights1[:, [d]][..., e_mask == 1], basis1, n_inputs=n_inputs_d)[:, 0]
        weights2_d = basis_invtransform_one_by_one(orig_weights2[:, [d]][..., e_mask == 1], basis2, n_inputs=n_inputs_d)[:, 0]
        fig, axs = plt.subplots(1, n_inputs_d, figsize=(3 * n_inputs_d, 3), sharey=sharey)

        stim = 0
        for __ in y_labels[_]:
            ax = axs[stim]

            for z in range(num_states):
                if z in skip_states:
                    continue
                ax.plot(np.arange(-weights1_d[z, stim].shape[-1], 0)/data_config1['orig_fps'], weights1_d[z, stim], color=COLORS[z], linewidth=3, label=f'State {z+1}')
                ax.plot(np.arange(-weights2_d[z, stim].shape[-1], 0)/data_config2['orig_fps'], weights2_d[z, stim], color=COLORS[z], linewidth=3, alpha =0.5, label=f'State {z+1}')

            ax.set_title(input_labels[__])

            ax.axhline(0, ls=':', c='k', lw=0.5)
            maxtime = weights1_d[0, stim].shape[-1]//data_config1['orig_fps']
            ax.set_xticks(np.linspace(-maxtime, 0, num=maxtime+1), labels=[-maxtime] + ['']*(maxtime-1) + [0])
            ax.margins(y=0.05)
            if saveindividual:
                ax.yaxis.set_tick_params(labelleft=True)  # ensure ytick labels are shown
            if ax.get_subplotspec().is_last_row() or saveindividual:
                ax.set_xlabel('Time (s)')
            stim += 1

        plt.margins(0.05)
        plt.tight_layout()
        fig.canvas.draw()

        if savefig:
            fig.savefig(os.path.join(fig_dir, f'{filesuffix}_filters_skip_states={skip_states}_sharey={sharey}_separate_emissions_{d}_{_}.pdf'), bbox_inches='tight', dpi=300, transparent=True)

        if saveindividual:
            # Save individual subplots
            os.makedirs(os.path.join(fig_dir, f'{filesuffix}_individual_filters'), exist_ok=True)
            stim = 0
            for __ in y_labels[_]:
                ax = axs[stim]
                extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(os.path.join(fig_dir, f'{filesuffix}_individual_filters', f'skip_states={skip_states}_subplot_{_}_{__}.pdf'), dpi=300, bbox_inches=extent, transparent=True)
                stim += 1

        if display: plt.show()
        plt.close()
    return


def plot_statetrans_filters_separate(orig_weights, data_config, input_list, input_mask_by_statetrans, input_labels, filesuffix, sharey=False, saveindividual=False, savefig=False, fig_dir=None, display=True):

    num_states = orig_weights.shape[0]
    print("orig_weights.shape", orig_weights.shape)     # (num states, num states, n_inputs * ncos), e.g. (5, 5, 84)

    filter_len = orig_weights.shape[-1]

    assert data_config['ncos'] == filter_len // len(data_config['input_labels_list'])

    basis = data_config['basis']
    print(orig_weights[..., input_mask_by_statetrans == 1].shape)
    weights = basis_invtransform_one_by_one(orig_weights[..., input_mask_by_statetrans == 1], basis, n_inputs=len(input_list))
    print("weights.shape after tr", weights.shape)

    for z_ in range(num_states):
        print("State", z_+1)

        n_inputs_z_ = len(input_list)
        weights_z_ = weights[z_]    # (5, 7, 450)
        print("weights_z_.shape", weights_z_.shape)
        fig, axs = plt.subplots(2, n_inputs_z_-2, figsize=(3 * (n_inputs_z_-2), 3+3), sharey=sharey)  # Split in 2 rows

        stim = 0
        i, j = 0, 0
        for __ in input_list:
            ax = axs[i][j]

            for z in range(num_states):
                if z == z_:
                    continue
                ax.plot(np.arange(-weights_z_[z, stim].shape[-1], 0)/data_config['orig_fps'], weights_z_[z, stim], color=COLORS[z], linewidth=3, label=f'To State {z+1}')

            ax.set_title(input_labels[__])
            ax.axhline(0, ls=':', c='k', lw=0.5)
            maxtime = weights_z_[0, stim].shape[-1]//data_config['orig_fps']
            ax.set_xticks(np.linspace(-maxtime, 0, num=maxtime+1), labels=[-maxtime] + ['']*(maxtime-1) + [0])
            ax.margins(y=0.05)
            if saveindividual:
                ax.yaxis.set_tick_params(labelleft=True)  # ensure ytick labels are shown
            if ax.get_subplotspec().is_last_row() or saveindividual:
                ax.set_xlabel('Time (s)')
            if ax.get_subplotspec().is_last_col():
                # ax.legend(loc='upper right')
                ax.legend(loc='upper right', bbox_to_anchor=(3, 1), borderaxespad=0.)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel('filter amplitude')
            stim += 1
            j = stim % 5
            i = stim // 5

        while i < 2:
            while j < 5:
                ax = axs[i][j]
                ax.axis('off')
                j += 1
            i += 1

        fig.supylabel(f'State transition filters\nFROM State {z_+1}', ha='center')
        fig.align_ylabels()
        plt.margins(0.05)
        plt.tight_layout()
        fig.canvas.draw()

        if savefig:
            fig.savefig(os.path.join(fig_dir, f'{filesuffix}_filters_sharey={sharey}_separate_state{z_+1}.pdf'), bbox_inches='tight', dpi=300, transparent=True)

        if display: plt.show()
        plt.close()
    return


def plot_statetrans_filters_separate_individual(orig_weights, data_config, input_list, input_mask_by_statetrans, input_labels, filesuffix, sharey=False, savefig=False, fig_dir=None, display=True):

    num_states = orig_weights.shape[0]
    print("orig_weights.shape", orig_weights.shape)     # (num states, num states, n_inputs * ncos), e.g. (5, 5, 84)

    filter_len = orig_weights.shape[-1]

    assert data_config['ncos'] == filter_len // len(data_config['input_labels_list'])

    basis = data_config['basis']
    print(orig_weights[..., input_mask_by_statetrans == 1].shape)
    weights = basis_invtransform_one_by_one(orig_weights[..., input_mask_by_statetrans == 1], basis, n_inputs=len(input_list))
    print("weights.shape after tr", weights.shape)

    for z_ in range(num_states):
        print("State", z_+1)

        weights_z_ = weights[z_]    # (5, 7, 450)

        stim = 0
        for __ in input_list:
            
            fig = plt.figure(figsize=(3.2, 3.2))
            ax = plt.gca()

            for z in range(num_states):
                if z == z_: continue
                ax.plot(np.arange(-weights_z_[z, stim].shape[-1], 0)/data_config['orig_fps'], weights_z_[z, stim], color=COLORS[z], linewidth=3, label=f'To State {z+1}')

            ax.set_title(input_labels[__])
            ax.axhline(0, ls=':', c='k', lw=0.5)
            maxtime = weights_z_[0, stim].shape[-1]//data_config['orig_fps']
            ax.set_xticks(np.linspace(-maxtime, 0, num=maxtime+1), labels=[-maxtime] + ['']*(maxtime-1) + [0])
            ax.margins(y=0.05)
            ax.yaxis.set_tick_params(labelleft=True)  # ensure ytick labels are shown
            ax.set_xlabel('Time (s)')
            # ax.set_ylabel('filter amplitude')
            ax.set_ylim([1.1 * np.min(weights_z_), 1.1 * np.max(weights_z_)])
            stim += 1
            plt.tight_layout()

            if savefig:
                fig.savefig(os.path.join(fig_dir, f'subplot_{z_+1}_{__}.pdf'), dpi=300, bbox_inches='tight', transparent=True)
            if display: plt.show()
            plt.close()
    return


def plot_statetrans_filter_amplitudes(orig_weights, data_config, input_list, input_mask_by_statetrans, input_labels, prefix='', savefig=False, fig_dir=None, display=True):
    # print(weights.shape)

    num_states = orig_weights.shape[0]
    print("orig_weights.shape", orig_weights.shape)  # (num states, num states, n_inputs * ncos), e.g. (5, 5, 84)

    filter_len = orig_weights.shape[-1]

    assert data_config['ncos'] == filter_len // len(data_config['input_labels_list'])

    basis = data_config['basis']
    print(orig_weights[..., input_mask_by_statetrans == 1].shape)   # (5, 5, 28)
    weights = basis_invtransform_one_by_one(orig_weights[..., input_mask_by_statetrans == 1], basis, n_inputs=len(input_list))
    print("weights.shape after tr", weights.shape)  # (num states, num states, n_inputs, 450), e.g.  (5, 5, 7, 450)

    fig_width = 3*num_states + 0.5
    fig, axs = plt.subplots(num_states, num_states-1, figsize=(fig_width, fig_width), sharey=True)

    xticklabels = np.array([input_labels[i] for i in input_list])
    for z_ in range(num_states):    # From State z_
        print("State", z_ + 1)

        weights_z_ = weights[z_]
        zi = 0
        for z in range(num_states):    # To State z
            if z == z_:
                continue
            w_l2 = np.linalg.norm(weights_z_[z], axis=-1)
            print("w_l2.shape", w_l2.shape)
            w_l2_scaled = (w_l2 - np.min(w_l2))/(np.max(w_l2) - np.min(w_l2) + 1e-8)
            sorted_idxs = np.argsort(w_l2_scaled)

            ax = axs[z_][zi]
            ax.plot(w_l2_scaled[sorted_idxs], 'k.', markersize=12)
            ax.set_xticks(range(len(sorted_idxs)), xticklabels[sorted_idxs], rotation=45, ha='right', rotation_mode='anchor')
            ax.set_yticks([0, 0.5, 1], [0, '', 1])
            ax.set_ylim([-0.1, 1.1])
            ax.margins(x=0.1)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.tick_params(axis='both', direction='in', top=True, right=True)
            ax.set_title(f'State {z + 1}', color=COLORS[z])
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(f'State {z_+1}', color=COLORS[z_])
            zi += 1

    fig.supylabel('Relative filter amplitude')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{prefix}_filter_amplitudes.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_filter_amplitudes(weights, data_config, input_labels, y_labels, input_mask_by_emission, prefix, skip_states=[], plot_top_k=None, savefig=False, fig_dir=None, display=True):
    # print(weights.shape)

    num_states = weights.shape[0]
    emission_dim = weights.shape[1]
    # filter_len = weights.shape[-1]
    # input_labels_list = data_config['input_labels_list']
    # input_labels = data_config['input_labels']
    # emission_labels_jr = data_config['emission_labels_jr']
    # y_labels = data_config['emission_labels']
    # n_inputs = len(input_labels_list)
    basis = data_config['basis']

    print("weights.shape", weights.shape)   # (num states, <emission_dim>, transformed filterlen), e.g. (5, 4, 84)
    # <emission_dim> is 4 here coz locomotion + wingflick

    # print(weights.shape)

    # input_mask_by_emission = data_config['input_mask_by_emission']

    eff_num_states = num_states-len(skip_states)

    if plot_top_k:
        fig_width = 3*eff_num_states + 0.5
    else:
        fig_width = 4*eff_num_states + 0.5
    fig, axs = plt.subplots(emission_dim, eff_num_states, figsize=(fig_width, 2.7*len(y_labels)), sharey=True)

    for d, _ in enumerate(y_labels):
        e_mask = input_mask_by_emission[d]
        print("y_labels[_]", _, y_labels[_])
        xticklabels = np.array([input_labels[i] for i in y_labels[_]])
        weights_d = weights[:, [d]][..., e_mask == 1]
        print("weights_d pre basistr", weights_d.shape)     # (num states, 1, transformed filterlen), e.g. (5, 1, 28)
        weights_d = basis_invtransform_one_by_one(weights_d, basis, n_inputs=len(y_labels[_]))[:, 0]
        print("weights_d post basistr", weights_d.shape)    # (num states, n_inputs, orig filterlen), e.g. (5, 7, 450)
        s = 0
        for z in range(num_states):
            if z in skip_states:
                continue
            w_l2 = np.linalg.norm(weights_d[z], axis=-1)
            w_l2_scaled = (w_l2 - np.min(w_l2))/(np.max(w_l2) - np.min(w_l2) + 1e-8)
            # print(w_l2_scaled, w_l2.shape, w_l2_scaled.shape)
            sorted_idxs = np.argsort(w_l2_scaled)
            if plot_top_k:
                k = min(5, len(sorted_idxs))
                sorted_idxs = sorted_idxs[np.r_[-k:0]]  # highest weighted inputs only
            if (eff_num_states > 1) and (emission_dim > 1):
                ax = axs[d, s]
            elif (eff_num_states == 1) and (emission_dim > 1):
                ax = axs[d]
            elif (eff_num_states > 1) and (emission_dim == 1):
                ax = axs[s]
            elif (eff_num_states == 1) and (emission_dim == 1):
                ax = axs
            ax.plot(w_l2_scaled[sorted_idxs], 'k.', markersize=12)
            ax.set_xticks(range(len(sorted_idxs)), xticklabels[sorted_idxs], rotation=45, ha='right', rotation_mode='anchor')
            ax.set_yticks([0, 0.5, 1], [0, '', 1])
            ax.set_ylim([-0.1, 1.1])
            ax.margins(x=0.1)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.tick_params(axis='both', direction='in', top=True, right=True)
            if ax.get_subplotspec().is_first_row() and prefix != 'aux_emissions':
                ax.set_title(f'State {z + 1}', color=COLORS[z])
            # if ax.get_subplotspec().is_first_col():
            #     ax.set_ylabel(emission_labels_jr[_], color=EC)
            s += 1

    # fig.supylabel('Relative filter amplitude')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{prefix}_filter_amplitudes_skip_states={skip_states}_plot_top_k={plot_top_k}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_o_dists(emissions_z, o_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(1, len(o_labels), figsize=(16, 5))

    for o, ol in enumerate(o_labels):
        x99 = 0
        x0 = 0
        ax = axes[o] if len(o_labels) > 1 else axes
        for z in list(emissions_z.keys()):
            data = np.random.choice(np.round(emissions_z[z][:, o], decimals=3), min(10000, len(emissions_z[z])), replace=False)
            x0 = min(x0, np.percentile(data, 2))
            x99 = max(x99, np.percentile(data, 98))
            sns.kdeplot(data, color=COLORS[z], ax=ax,
                        # cumulative=True,
                         common_norm=True,
                         # kde=True,
                         # stat='probability',
                         label=f'State {z+1}',
                         # edgecolor=None,
                         alpha=1,
                        cut=0,
                        linewidth=2,
                         # bins=100,
                        clip=(x0, x99)
                        )
        ax.axvline(0, lw=0.5, c='gray', ls=':', alpha=0.7)
        # ax.set_xscale('symlog')  # symmetric log, can handle negative emission values with log_scale in sns.histplot can't.
        # ax.set_yscale('log')
        ax.set_xlabel(o_labels[ol], color=EC)
        # ax.margins(y=0.1)
        ax.set_xlim([x0, x99])
        ax.set_ylim([0, 2])
        # ax.axhline(0, ls=":", lw=2, c='k')

    ax.legend(loc='upper right')
    # fig.suptitle(f'behavioral outputs by state')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_mean_outputs_by_o_dists.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_o_dists_reformatted(emissions_z, o_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(1, len(o_labels), figsize=(16, 5))

    def reformat(f_name, dt):
        if f_name in ['fFV', 'fFA', 'mFV']:
            t = dt
            xlim = (-1.5, 4.5)
            ylim = (0, 2)
        elif f_name in ['fLV', 'fLA', 'mLV']:
            t = dt
            xlim = (-2.5, 2.5)
            ylim = (0, 2)
        elif f_name in ['fAV', 'fAA', 'mAV']:
            t = dt
            xlim = (-110, 110)
            ylim = (0, 0.1)
        else:
            raise Exception(f'Unsupported o feat: {f_name}.')
        return t, xlim, ylim

    for o, ol in enumerate(o_labels):
        ax = axes[o] if len(o_labels) > 1 else axes
        for z in list(emissions_z.keys()):
            data_z = emissions_z[z][:, o]
            data_z_reformatted, xlim, ylim = reformat(ol, data_z)
            samples = np.random.choice(np.round(data_z_reformatted, decimals=3), min(10000, len(data_z_reformatted)), replace=False)
            sns.kdeplot(samples, color=COLORS[z], ax=ax,
                        common_norm=True,
                        label=f'State {z+1}',
                        alpha=1,
                        cut=0,
                        linewidth=2,
                        bw_adjust=2,
                        )
        # ax.axvline(0, lw=0.5, c='gray', ls=':', alpha=0.7)
        ax.set_xlabel(o_labels[ol], color=EC)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # ax.legend(loc='upper right')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_mean_outputs_by_o_dists_reformatted.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_o_dists_otherfilters(all_soft_predictions_per_state, all_stateseq, num_states, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    if num_states <= 1:
        return

    emission_dim = len(o_labels)
    n_batches = len(all_soft_predictions_per_state)

    fig, axes = plt.subplots(emission_dim, num_states, figsize=(3*num_states+0.5, 3*emission_dim), sharex=True, sharey='row')

    emissions_using_z_in_z_ = {}
    for z_ in range(num_states):        # In state z_
        state_masks_z_ = [all_stateseq[btch] == z_ for btch in range(n_batches)]   # get mask for each session where state == z_
        print(f"In state {z_}:")

        emissions_using_z_in_z_[z_] = {}
        for z in range(num_states):     # using filters from z
            emissions_using_z_in_z_[z_][z] = np.vstack([all_soft_predictions_per_state[btch][state_masks_z_[btch], z, :] for btch in range(n_batches)])  # get predictions using z when state == z_
            print(f"using filters from {z}:", np.mean(emissions_using_z_in_z_[z_][z], axis=0))

    for z_ in range(num_states):
        for o, ol in enumerate(o_labels):
            ax = axes[o, z_]
            for z in range(num_states):
                data = emissions_using_z_in_z_[z_][z][:, o]
                print(z_, ol, np.mean(data), np.median(data))

                min_x, max_x = np.percentile(data, 2), np.percentile(data, 98)
                data_filtered = data[(data >= min_x) & (data <= max_x)]
                data_filtered = np.random.choice(data_filtered, size=min(100000, len(data_filtered)), replace=False)
                sns.violinplot(x=z, y=data_filtered, ax=ax, color=COLORS[z],
                               fill=False,
                               inner='quartile',  # shows IQR only (no scatter/sticks)
                               cut=0,  # do not extend beyond min/max of data
                               density_norm='area',  # makes violins comparable
                               linewidth=2  # remove outline
                               # common_norm=True,
                               )
            ax.set_xticks(range(num_states), [f'State {s+1}' for s in range(num_states)], rotation=45, ha='right', rotation_mode='anchor')
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(o_labels[ol])
            if ax.get_subplotspec().is_first_row():
                ax.set_title(f'State {z_+1}', color=COLORS[z_])
    # fig.supylabel('Density')
    fig.align_ylabels(axes[:, 0])
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_odists_otherfilters.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_aux_dists(aux_z, a_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, ax = plt.subplots(1, len(a_labels), figsize=(25, 4))

    for a, al in enumerate(a_labels):
        x99 = 0
        x0 = 0
        for z in list(aux_z.keys()):
            data = np.random.choice(np.round(aux_z[z][:, a], decimals=3), min(10000, len(aux_z[z][:, a])), replace=False)
            x0 = min(x0, 2*np.percentile(data, 0.1))
            x99 = max(x99, np.percentile(data, 99.5))
            # print(al, x0, x99)
            sns.kdeplot(data, color=COLORS[z], ax=ax[a],
                        # common_norm=True,
                        label=f'State {z+1}',
                        alpha=1,
                        # cut=1,
                        linewidth=2,
                        clip=(x0, x99),
                        )

        ax[a].axvline(0, lw=0.5, c='gray', ls=':', alpha=0.7)
        ax[a].set_xlabel('z-score', color='k')
        ax[a].set_title(a_labels[al])
        # ax[a].set_yscale('log')
        ax[a].margins(y=0.1,x=0.1)
        # ax[a].set_xlim([x0-0.1, x99+0.1])
        ax[a].set_xlim(-4, 4)
        ax[a].set_xticks([-4, -2, 0, 2, 4])
        # ymin, ymax = ax[a].get_ylim()
        # ax[a].set_ylim(ymin-0.01*ymax, ymax)
        # ax[a].axhline(0, ls=":", lw=2, c='k')
    # ax[-1].legend(loc='upper right')
    # fig.suptitle(f'Sensory inputs by state')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_mean_aux_dists.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_aux_dists_reformatted(aux_z, a_labels, config, exclude_a=[], title=None, savefig=False, fig_dir=None, display=True):

    def reformat(f_name, dt):
        # print(f_name)
        axtitle = None
        if f_name in ['mFV', 'mFS', 'fFV', 'fFS']:
            t = dt * config['effective_fps']
            cut = 0
            xlabel = '(mm/s)'
            ylabel = 'Density'
            xlim = (-0.25, 1.25)
            xticks = [0, 0.5, 1,]
        elif f_name in ['mLS', 'fLS']:
            t = dt * config['effective_fps']
            cut = 0
            xlabel = '(mm/s)'
            ylabel = 'Density'
            xlim = (-0.03, 1.25)
            xticks = [0, 0.5, 1,]
        elif f_name in ['mfDist']:
            t = dt
            cut = 1
            xlabel = '(mm)'
            ylabel = 'Density'
            xlim = (0, 10)
            xticks = [0, 5, 10]
        elif f_name in ['fmAng_cos']:
            t = np.rad2deg(np.arccos(dt))
            cut = 0
            xlabel = '(deg)'
            ylabel = 'Density'
            xlim = (-2, 182)
            xticks = [0, 90, 180]
            axtitle = '|fmAng|'
        elif f_name in ['fmAng_sin']:
            t = np.rad2deg(np.arcsin(dt))
            cut = 0
            xlabel = '(deg)'
            ylabel = 'Density'
            xlim = (-92, 92)
            xticks = [-90, 0, 90]
            axtitle = 'male lateral position'
        elif f_name in ['mfAng_cos']:
            t = np.rad2deg(np.arccos(dt))
            cut = 0
            xlabel = '(deg)'
            ylabel = 'Density'
            xlim = (-2, 182)
            xticks = [0, 90, 180]
            axtitle = '|mfAng|'
        elif f_name in ['mfAng_sin']:
            t = np.rad2deg(np.arcsin(dt))
            cut = 0
            xlabel = '(deg)'
            ylabel = 'Density'
            xlim = (-92, 92)
            xticks = [-90, 0, 90]
            axtitle = 'female lateral position'
        elif f_name in ['pulse_i', 'sine_i', 'tap2']:
            # print(f_name, np.unique(dt, return_counts=True))
            # print(f_name, np.unique(np.round(dt, 2), return_counts=True))
            # print(f_name, np.sum(dt), len(dt))
            t = np.sum(dt) / len(dt)
            cut = None
            xlabel = None
            ylabel = f'P({a_labels[f_name]}=1 | state)'
            xlim = None
            xticks = []
        elif f_name in ['wingAlign']:
            t = dt
            cut = 0
            xlabel = '(deg)'
            ylabel = 'Density'
            xlim = None
            xticks = [0, 90, 180]
            axtitle = 'wing align'
        else:
            raise Exception(f'Unsupported aux feature: {f_name}.')
        return t, cut, xlabel, ylabel, xlim, xticks, axtitle

    len_eff_a_labels = len([_ for _ in a_labels if _ not in exclude_a])
    fig, ax = plt.subplots(1, len_eff_a_labels, figsize=(3.5 * len_eff_a_labels, 4))

    axi = 0
    for a, al in enumerate(a_labels):
        print(al)
        if al in exclude_a:
            continue
        for z in list(aux_z.keys()):
            data_z = aux_z[z][:, a]
            data_z_reformatted, cut, xlabel, ylabel, xlim, xticks, axtitle = reformat(al, data_z)
            print(f'State {z+1}', np.nanmean(data_z_reformatted), xlabel)

            if al in ['pulse_i', 'sine_i', 'tap2']:
                ax[axi].bar(z/2, data_z_reformatted, color=COLORS[z], alpha=0.9, width=0.3)
            else:
                samples = np.random.choice(data_z_reformatted, min(10000, len(data_z)), replace=False)
                sns.kdeplot(samples, color=COLORS[z], ax=ax[axi],
                            label=f'State {z+1}',
                            alpha=1,
                            cut=cut,
                            linewidth=3,
                            )

        # ax[axi].axvline(0, lw=0.5, c='gray', ls=':', alpha=0.7)
        ax[axi].set_title(axtitle if axtitle else a_labels[al])
        ax[axi].set_xlabel(xlabel)
        ax[axi].set_ylabel(ylabel)
        ax[axi].set_xlim(xlim)
        ax[axi].set_xticks(xticks)
        ax[axi].margins(y=0.1,x=0.1)
        # ax[axi].title.set_position([.5, 10.05])
        axi += 1
    fig.align_xlabels()
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_mean_aux_dists_exclude_a={len(exclude_a)}_reformatted.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_aux_sorted_odists(aux_z, emissions_z, a_labels, o_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(1, len(aux_z.keys())*2, figsize=(len(aux_z.keys())*5+0.1, 10), gridspec_kw={'width_ratios': [2, 1]*len(aux_z.keys())})

    for z in list(aux_z.keys()):
        aux_means = np.mean(aux_z[z], axis=0)
        # max_y = np.max([max_y, np.max([*np.abs(aux_means), *np.abs(o_means)]) + 0.1])
        max_y = np.max(np.abs(aux_means)) + 0.1
        print("max_y", max_y)

        ax_col1, ax_col2 = 2*z, 2*z+1

        ax = axes[ax_col1]
        v = aux_means
        sorted_by = np.argsort(np.abs(v))[::-1]
        values = v[sorted_by]
        ax.plot([-1, len(values)], [0, 0], color='k', lw=1)
        ax.bar(range(len(values)), values, color=COLORS[z], alpha=0.6)
        ax.margins(0.1)
        ax.set_title(f'State {z+1}\n\n', color=COLORS[z], loc='right')
        ax.axis('off')
        arrow_max = max_y
        ax.plot([-1, -1], [-arrow_max, arrow_max], color='k', lw=3)
        ax.add_patch(FancyArrowPatch((-1, arrow_max), (-1, arrow_max+0.05), arrowstyle="->", mutation_scale=35, linewidth=3, color='k'))
        ax.add_patch(FancyArrowPatch((-1, -arrow_max), (-1, -arrow_max-0.05), arrowstyle="->", mutation_scale=35, linewidth=3, color='k'))
        for x, label in zip(range(len(values)), np.array(list(a_labels.values()))[sorted_by]):
            if np.sign(values[x]) >= 0:
                ax.text(x, 0.01, label, ha='center', va='bottom', rotation=90, fontsize=16)
            else:
                ax.text(x, -0.01, label, ha='center', va='top', rotation=90, fontsize=16)
        ax.text(-1.7, arrow_max-arrow_max*0.1, 'More', ha='center', va='bottom', rotation=90, fontsize=18)
        ax.text(-1.7, -arrow_max+arrow_max*0.1, 'Less', ha='center', va='bottom', rotation=90, fontsize=18)
        ax.text(-1.7, 0, 'Relative z-scores', ha='center', va='center', rotation=90, fontsize=18)
        ax.set_ylim(-max_y-0.05, max_y+0.05)

        ax = axes[ax_col2]
        for o, ol in enumerate(o_labels):
            print(z, ol, np.mean(emissions_z[z][:, o]), np.median(emissions_z[z][:, o]))

            data = emissions_z[z][:, o]
            min_x, max_x = np.percentile(data, 2), np.percentile(data, 98)
            data_filtered = data[(data >= min_x) & (data <= max_x)]
            data_filtered = np.random.choice(data_filtered, size=min(100000, len(data_filtered)), replace=False)

            sns.violinplot(x=o, y=data_filtered, ax=ax, color=COLORS[z],
                           fill=False,
                           inner='quartile',  # shows IQR only (no scatter/sticks)
                           cut=0,  # do not extend beyond min/max of data
                           density_norm='area',  # makes violins comparable
                           linewidth=2  # remove outline
                           # common_norm=True,
                           )
        ax.axhline(0, c='k', linewidth=1, ls='-')
        ax.set_xticks(range(len(o_labels)))
        ax.set_xticklabels(np.array(list(o_labels.values())), rotation=90, color=EC)
        ymin, ymax = ax.get_ylim()
        max_abs = max(abs(ymin), abs(ymax)) + 0.05
        max_abs = np.round(max_abs) if max_abs > 1 else np.round(max_abs, 1)
        ax.set_ylim(-max_abs, max_abs)
        print(ymin, ymax)
        ax.set_yticks([-max_abs, 0, max_abs])

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_zscored_mean_aux_fullodists_sorted.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_dwell_times(dwell_times_z, num_states, effective_fps, title='', savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()

    durations = []
    for z in range(num_states):
        d = dwell_times_z[z] / effective_fps    # in seconds
        print(f"State {z + 1}: Mean dwell time = {np.mean(d)*1000:.2f}ms and STD = {np.std(d)*1000:.2f}ms")
        durations.append(d.tolist())

    df = pd.DataFrame({
        "dwell_time": sum(durations, []),
        "state": sum([[z] * len(v) for z, v in enumerate(durations)], [])
    })
    state_labels = dict([(z, f'State {z+1}') for z in np.arange(num_states)])
    palette = dict([(f'State {z+1}', COLORS[z]) for z in np.arange(num_states)])
    df["state_label"] = df["state"].map(state_labels)

    sns.kdeplot(
        ax=ax,
        data=df,
        x="dwell_time",
        hue="state_label",
        palette=palette,
        bw_adjust=3,
        cut=0,
        # clip=(0, None),
        # common_norm=True,
        linewidth=3,
        legend=False,
    )

    ax.set_xscale('log')

    plt.xlabel("Time (s)")
    plt.title("State residency")
    ylim = plt.ylim()
    print("ylim", ylim)
    plt.ylim(ylim[0]-0.2, ylim[1]+0.2)

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'state_dwell_times.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_prob_states_aligned(resampled_state_seq, n_le=None, uniform_filter_size=1, config=None, title=None, xticks=None, xlabel=None, savefig=False, fig_dir=None, display=True):

    fig = plt.figure(figsize=(9, 4))

    GRID = resampled_state_seq.shape[1]     # reshaped timesteps
    percent = np.linspace(0, GRID-1, GRID)

    for z in range(config['num_states']):
        if n_le is None:
            mean = resampled_state_seq[:, :, z].mean(0)
            sem = resampled_state_seq[:, :, z].std(0, ddof=1) / np.sqrt(len(resampled_state_seq))
        else:
            mean = np.sum(resampled_state_seq[:, :, z], axis=0)/n_le
            # sem = np.sqrt(((resampled_state_seq[:, :, z].sum(0) - mean)**2) / n_le) / np.sqrt(n_le)

        # plt.plot(percent, mean, '-', c=COLORS[z], lw=2, label=f'State {z+1}')
        plt.plot(percent, uniform_filter1d(mean, size=uniform_filter_size), '-', c=COLORS[z], lw=2, label=f'State {z+1}')
        # plt.fill_between(percent, mean - sem, mean + sem, alpha=.3)

    # plt.grid(alpha=.3)
    plt.ylim(0, 0.6)
    plt.xlabel(xlabel)
    plt.ylabel('p(state)')
    # plt.legend(loc='upper left')
    # if title: plt.title(f'{title} sessions')
    if xticks: plt.xticks([0, GRID-1], xticks)
    # plt.yticks(ticks=np.linspace(0, 0.7, num=8), labels=[0] + ['']*6 + [0.7])
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_prob_states_over_time.pdf'), bbox_inches='tight', dpi=300,
                            transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_transition_matrix(transition_matrix, title=None, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    m = transition_matrix.shape[0]
    sns.heatmap(transition_matrix, annot=True, cmap='bone', cbar=False, square=True, fmt=".2f",
                vmin=0, vmax=1, ax=ax,
                xticklabels=[f'{i+1}' for i in range(m)],
                yticklabels=[f'{i+1}' for i in range(m)], annot_kws={'size': 'small'})
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(length=0)
    # plt.title('Transition Matrix')
    plt.xlabel('state t')
    plt.ylabel('state t-1')
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = title if title else 'transition_matrix'
    if savefig: fig.savefig(os.path.join(fig_dir, f'{filename}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_ethogram(transition_matrix, title=None, savefig=False, fig_dir=None, display=True):
    print(transition_matrix.tolist())
    fig = plt.figure()

    G = nx.DiGraph()
    num_states = transition_matrix.shape[0]

    # Add edges with weights
    for i in range(num_states):
        for j in range(num_states):
            if round(transition_matrix[i, j], 2) > 0.01:  # Only add edges with nonzero probability
                G.add_edge(i, j, weight=transition_matrix[i, j])

    pos = nx.spring_layout(G)
    # nx.draw_networkx
    # print("nodes", list(G), G.edges, G.nodes)

    edges = G.edges(data=True)
    edge_widths = [d['weight'] * 5 for (u, v, d) in edges]  # Scale edge width
    nx.draw(G, pos,
            with_labels=True,
            node_color=[COLORS[_] for _ in G.nodes],
            labels={_:_+1 for _ in G.nodes},
            node_size=1000,
            font_size=15, font_weight='bold',
            edge_color='black', width=edge_widths,
            arrows=True,
            connectionstyle='arc3,rad=0.4'
            )

    # Draw edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for (u, v, d) in edges}
    # print(edge_labels, len(edge_labels))
    nx.draw_networkx_edge_labels(G, pos, font_size=15, edge_labels=edge_labels, label_pos=0.8, rotate=False,
                                 # connectionstyle='arc3,rad=0.4'
                                 )

    # plt.title("Transition Probability Graph")
    plt.tight_layout()
    plt.margins(0.1)
    filename = title if title else 'ethogram'
    if savefig: fig.savefig(os.path.join(fig_dir, f'{filename}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_expected_occupancy(steady_state_p, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(5, 5))
    # m = steady_state_p.shape[0]
    # sns.heatmap(steady_state_p[None, :], annot=True, cmap='bone', cbar=False, square=True,
    #             fmt=".2f",
    #             xticklabels=[f'State {_}' for _ in range(m)])
    # plt.yticks([])
    # plt.title('Steady State Probabilities')

    print("steady_state_p", steady_state_p)

    for z in range(len(steady_state_p)):
        plt.bar(z+1, steady_state_p[z], color=COLORS[z])
    plt.ylabel('Fraction occupancy')
    plt.margins(0.1)

    pmax = np.round(np.max(steady_state_p), 1) + 0.2
    plt.ylim(0, pmax)
    plt.yticks([0, pmax/2, pmax])
    plt.xticks(range(1, 1+len(steady_state_p)))
    plt.xlabel('State')
    # plt.title('Expected long-run state occupancy')
    if savefig: fig.savefig(os.path.join(fig_dir, 'expected_occupancy.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_empirical_occupancy(state_seqs, config, title='', savefig=False, fig_dir=None, display=True):
    from collections import defaultdict
    ps_z = defaultdict(list)
    for i in range(len(state_seqs)):
        state_z, count_z = np.unique(state_seqs[i], return_counts=True)
        # print(i, state_z, count_z)
        percent_z = count_z / np.sum(count_z)
        s_p_dict = dict(zip(state_z, percent_z))
        for z in range(config['num_states']):
            ps_z[z].append(s_p_dict.get(z, 0))

    sort_by = np.argsort(ps_z[0])[::-1]
    n_colors = len(ps_z[0])
    colors = np.linspace(0, 1, n_colors)

    pmax = -1

    fig = plt.figure(figsize=(5, 5))
    for z in ps_z:

        ps_z[z] = np.array(ps_z[z])
        base_rgba = to_rgba(COLORS[z], alpha=1.0)
        faded_colors = np.tile(base_rgba, (n_colors, 1))
        faded_colors[:, -1] = np.linspace(1, 0.3, n_colors)
        transparent_cmap = ListedColormap(faded_colors)

        jitter = np.random.uniform(-0.1, 0.1, len(ps_z[z]))
        plt.scatter(z+1+jitter, ps_z[z][sort_by], c=colors, cmap=transparent_cmap, s=SCATTERSIZE, edgecolors='none')
        plt.errorbar(z+1+.2, np.mean(ps_z[z]), yerr=np.std(ps_z[z]), color='k', alpha=0.8, fmt='o', capsize=0)
        pmax = max(pmax, np.max(ps_z[z]))

    plt.ylabel('Fraction occupancy')
    plt.margins(0.1)
    pmax = np.round(pmax, 1) + 0.2
    plt.ylim(0, pmax)
    plt.yticks([0, pmax / 3, 2*pmax/3, pmax])
    plt.xticks(range(1, 1 + len(ps_z)))
    plt.xlabel('State')
    # plt.title(f'Empirical state occupancy')
    if savefig: fig.savefig(os.path.join(fig_dir, f'empirical_occupancy_{title.lower()}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_loss(em_losses, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    plt.plot(em_losses, '.-')
    # print("em_losses:", em_losses)
    plt.title('EM training iters')
    plt.xlabel('#iters')
    plt.ylabel('Neg Log prob (per frame)')
    plt.margins(0.2)
    if savefig: fig.savefig(os.path.join(fig_dir, 'loss.pdf'), bbox_inches='tight', dpi=300)
    if display: plt.show()
    plt.close()
    return fig


def plot_state_probs(state_probs, model_config, data_config, batch, effective_fps, xlim=None, xlim_orig=None, prefix='', suffix='', savefig=False, fig_path=None, display=True):
    xlim_ = np.r_[xlim[0]:xlim[1]+1]

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    for z in range(model_config['num_states']):
        plt.plot(xlim_, state_probs[batch][xlim_, z], c=COLORS[z], linewidth=3, label=f'State {z+1}')

    plt.ylim([-0.05, 1.05])
    plt.yticks([0, 1])

    xt = np.linspace(xlim_[0], xlim_[-1], num=5)
    pws = data_config['predict_window_size']
    init_period = data_config['input_raw_each_dim']
    orig_fps = data_config['orig_fps']
    ax.set_xticks(xt)
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.set_xticklabels([f"{round((pws * x + init_period) / orig_fps, 1)}" for x in xt])

    plt.ylabel('P(state | data)')
    # plt.title(f'{prefix.title()}:{batch} {suffix}')
    plt.xlabel('Time (s)')
    # plt.legend(loc='upper right')

    plt.tight_layout()
    if savefig: fig.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_trajectories(model_ckp, model_config, data_config, batch, prefix='', suffix='', states_in_bgr=True, xlim=None, xlim_orig=None, savefig=False, fig_path=None, display=True):

    # num_states = model_config['num_states']
    model_label = model_ckp['prefix'].upper() #+ '_' + str(num_states)
    emission_labels_jr = data_config['emission_labels_jr']

    emissions = model_ckp[f'{prefix}_data'][f'{prefix}_soft_predictions'][batch]
    true_emissions = model_ckp[f'{prefix}_data'][f'{prefix}_emissions'][batch]
    stateseq = model_ckp[f'{prefix}_data'][f'{prefix}_stateseq'][batch]

    # plot_hmm_data_whole_session_with_states(
    #     emissions, true_emissions, stateseq, data_config, model_label=model_label, y_labels=emission_labels_zscored, xlim=xlim, xlim_orig=xlim_orig,
    #     title=f'Predicted trajectory ({prefix.capitalize()}:{batch}) {suffix}',
    #     savefig=savefig, fig_path=fig_path, display=display)
    plot_hmm_data_whole_session_with_states_on_top(
        emissions, true_emissions, stateseq, data_config, model_label=model_label, y_labels=emission_labels_jr, xlim=xlim, xlim_orig=xlim_orig,
        title=f'Predicted trajectory ({prefix.capitalize()}:{batch}) {suffix}',
        savefig=savefig, fig_path=fig_path, display=display)
    return


def plot_trajectories_statewise(model_ckp, model_config, data_config, batch, prefix='', suffix='', states_in_bgr=True, xlim=None, xlim_orig=None, savefig=False, fig_path=None, display=True):
    # num_states = model_config['num_states']
    model_label = model_ckp['prefix'].upper() #+ '_' + str(num_states)
    # emission_labels = data_config['emission_labels_zs']
    emission_labels_zscored = data_config['emission_labels_zscored']

    emissions = model_ckp[f'{prefix}_data'][f'{prefix}_soft_predictions'][batch]
    emissions_per_state = model_ckp[f'{prefix}_data'][f'{prefix}_soft_predictions_per_state'][batch]
    # print("emissions_per_state", emissions_per_state.shape)
    true_emissions = model_ckp[f'{prefix}_data'][f'{prefix}_emissions'][batch]
    stateseq = model_ckp[f'{prefix}_data'][f'{prefix}_stateseq'][batch]

    for o_index in range(emissions.shape[-1]):
        plot_hmm_data_whole_session_perstate_states_on_top(
            emissions, emissions_per_state, true_emissions, stateseq, data_config, model_config, o_index=o_index, model_label=model_label, y_labels=emission_labels_zscored, xlim=xlim, xlim_orig=xlim_orig,
            savefig=savefig, fig_path=f'{fig_path}_o_index={str(o_index)}.pdf', display=display)
    return


def plot_traces_session(fTrx, mTrx, intervals, z, output_path=None, title=None, savefig=None, display=False):
    from preprocess.leaprig import WT_DATA
    DATA = WT_DATA

    fly_nodes = DATA.get_fly_nodes()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    grid_size = int(np.round(np.sqrt(len(intervals))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(4.5, 4.5)
                             # , sharex=True, sharey=True
                             )
    head_idx = fly_nodes.index('head')
    thorax_idx = fly_nodes.index('thorax')

    for i in range(grid_size):
        for j in range(grid_size):
            ax = axes[i, j]

            s, e = intervals[i * grid_size + j]

            xf_hd_tr, yf_hd_tr = fTrx[s:e][:, head_idx, 1], fTrx[s:e][:, head_idx, 0]
            xm_hd_tr, ym_hd_tr = mTrx[s:e][:, head_idx, 1], mTrx[s:e][:, head_idx, 0]
            # xf_thx_tr, yf_thx_tr = fTrx[s:e][:, thorax_idx, 1], fTrx[s:e][:, thorax_idx, 0]
            # xm_thx_tr, ym_thx_tr = mTrx[s:e][:, thorax_idx, 1], mTrx[s:e][:, thorax_idx, 0]

            traj1 = np.vstack([xf_hd_tr, yf_hd_tr])
            traj2 = np.vstack([xm_hd_tr, ym_hd_tr])

            combined = np.hstack([traj1, traj2])
            # print(combined.shape, traj1.shape, traj2.shape)
            x_center = np.nanmean(combined[0, :])
            y_center = np.nanmean(combined[1, :])

            buffer = 8

            ax.plot(xf_hd_tr, yf_hd_tr, '-', c=EC, linewidth=4, label='female')
            ax.plot(xm_hd_tr, ym_hd_tr, '-', c=IC, linewidth=3, label='male')
            # ax.plot(xf_thx_tr, yf_thx_tr, '.:', c=EC, linewidth=lw, alpha=0.1, label='female')
            # ax.plot(xm_thx_tr, ym_thx_tr, '.:', c=IC, linewidth=lw, alpha=0.1, label='male')

            ax.set_xlim(x_center - buffer, x_center + buffer)
            ax.set_ylim(y_center - buffer, y_center + buffer)

            arrow_e = e
            xf_thx, yf_thx = fTrx[arrow_e][thorax_idx, 1], fTrx[arrow_e][thorax_idx, 0]
            xm_thx, ym_thx = mTrx[arrow_e][thorax_idx, 1], mTrx[arrow_e][thorax_idx, 0]
            xf_hd, yf_hd = fTrx[arrow_e][head_idx, 1], fTrx[arrow_e][head_idx, 0]
            xm_hd, ym_hd = mTrx[arrow_e][head_idx, 1], mTrx[arrow_e][head_idx, 0]

            ax.add_patch(FancyArrowPatch((xf_thx, yf_thx), (xf_hd, yf_hd), arrowstyle='->', mutation_scale=15, color='k', linewidth=2, zorder=10))
            ax.add_patch(FancyArrowPatch((xm_thx, ym_thx), (xm_hd, ym_hd), arrowstyle='->', mutation_scale=15, color='k', linewidth=2, zorder=10))

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.margins(0.1)

            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_edgecolor('gray')
                ax.spines[spine].set_linewidth(2)
                ax.spines[spine].set_alpha(0.3)

    plt.suptitle(title, color=COLORS[z])
    plt.tight_layout()
    if savefig: fig.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return
