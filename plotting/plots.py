import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.ticker import FixedLocator
from matplotlib.patches import FancyArrowPatch, Rectangle
import seaborn as sns
import os
import networkx as nx
import pandas as pd

import numpy as np
# import jax.numpy as jnp
# import jax.random as jr
from dynamax.utils.plotting import CMAP, COLORS
from scipy.ndimage import uniform_filter1d
from glm_utils.preprocessing import BasisProjection
from sklearn.metrics import log_loss, balanced_accuracy_score, f1_score, recall_score, precision_score
from preprocess.colors import *
from utilities.io import get_feat_windows, get_event_onsets, basis_invtransform_one_by_one

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
    return

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


def plot_hmm_data(emissions, states, extra_data=None, xlim=None, y_labels=None, extra_data_labels=None, title=None):
    """Plot emissions vs. time with background colored by true state"""
    num_timesteps = len(emissions)
    emission_dim = emissions.shape[-1]
    extra_data_dim = extra_data.shape[-1] if extra_data is not None else 0

    # Plot the data superimposed on the generating state sequence
    fig, axs = plt.subplots(emission_dim+extra_data_dim, 1, sharex=True)

    d = 0
    for _ in y_labels:
        ax = axs[d]
        lim = 1.05 * abs(emissions[:, d]).max()
        ax.imshow(states[None, :], aspect="auto", interpolation="none", cmap=CMAP, vmin=0, vmax=len(COLORS) - 1,
                  extent=(0, num_timesteps, -lim, lim))
        # bounds = np.unique(states).astype(int)
        # norm = colors.BoundaryNorm(bounds, CMAP.N)
        # plt.colorbar(img, boundaries=bounds, ax=ax)
        # ax.legend(bounds, [f'State {s}' for s in bounds], loc='upper left')
        ax.plot(emissions[:, d], "-k")
        # axs[d].plot(means[:, d], ":k")
        if y_labels is not None:
            ax.set_ylabel(f'${{{y_labels[_]}_t}}$', fontsize='medium')
        d += 1
        # ax.xaxis.set_visible(False)
        # axs[d].set_ylabel("$y_{{t,{} }}$".format(d + 1))

    # for d in range(extra_data_dim):
    #     ax = axs[d+emission_dim]
    #     lim = 1.05 * abs(extra_data[:, d]).max()
    #     ax.imshow(states[None, :], aspect="auto", interpolation="none", cmap=CMAP,
    #                   vmin=0, vmax=len(COLORS) - 1,
    #                   extent=(0, num_timesteps, -lim, lim)
    #                   )
    #     ax.plot(extra_data[:, d], "-k")
    #     if extra_data is not None:
    #         ax.plot(extra_data[:, d], ":k")
    #     if extra_data_labels is not None:
    #         ax.set_ylabel(f'${{{extra_data_labels[d]}_t}}$')
    #     if d == 0:
    #         ax.set_title('Extra series not used for training model.')

    if xlim is None:
        plt.xlim(0, num_timesteps)
    else:
        plt.xlim(xlim)

    # axs[0].set_xticks([0, len(emissions[:, 0])], [0, 15], fontsize='medium')
    # axs[-1].set_xlabel("Time (min)")
    axs[0].set_title('State segmentation on one example session')
    plt.tight_layout()
    fig.align_ylabels()
    return fig


def plot_hmm_data3(sampled_emissions, true_emissions, states, xlim=None, y_labels=None, title=None):
    """Plot emissions vs. time with background colored by true state"""
    num_timesteps = len(sampled_emissions)
    emission_dim = sampled_emissions.shape[-1]

    # Plot the data superimposed on the generating state sequence
    fig, axs = plt.subplots(emission_dim, 1, sharex=True)

    d = 0
    for _ in y_labels:
        ax = axs[d]
        max_value = max(abs(sampled_emissions[:, d]).max(), abs(true_emissions[:, d]).max())
        lim = 1.05 * max_value
        ax.imshow(states[None, :], aspect="auto", interpolation="none", cmap=CMAP, vmin=0, vmax=len(COLORS) - 1,
                  extent=(0, num_timesteps, -lim, lim)
                  )

        ax.plot(true_emissions[:, d], "k-", label='True')
        ax.plot(sampled_emissions[:, d], "m-", label='Predicted')
        ax.set_ylabel(f'${{{y_labels[_]}_t}}$', fontsize='medium')
        d += 1

    plt.yticks([])

    if xlim is None:
        plt.xlim(0, num_timesteps)
    else:
        plt.xlim(xlim)

    # plt.xticks([0, len(sampled_emissions[:, 0])], [0, 15], fontsize='medium')
    plt.xlabel("Time", fontsize='medium')
    axs[0].legend(loc='upper left')
    plt.suptitle(title)
    fig.align_ylabels()
    return fig


def plot_hmm_data_whole_session_multiple_models(predicted_emissions, true_emissions, xlim=None, model_labels=None, y_labels=None, title=None):
    """Plot predicted and true emissions vs. time, for multiple models"""
    if predicted_emissions.ndim <= 2:
        predicted_emissions = np.expand_dims(predicted_emissions, 0)
    num_timesteps = len(predicted_emissions[1])
    emission_dim = predicted_emissions.shape[-1]
    fig, axs = plt.subplots(emission_dim, 1, figsize=(15, 10), sharex=True)
    d = 0
    for _ in y_labels:
        ax = axs[d]
        ax.plot(true_emissions[:, d], "k-", alpha=0.6, label='Data')
        for label, model_predicted_emissions in zip(model_labels, predicted_emissions):
            ax.plot(model_predicted_emissions[:, d], '-', linewidth=1.5, label=f'{label}')
        ax.set_ylabel(f'${{{y_labels[_]}_t}}$', fontsize='medium')
        d += 1
    if xlim is None:
        xlim = (0, num_timesteps)
    plt.xlim(xlim)
    plt.xlabel("Time", fontsize='medium')
    axs[0].legend(loc='upper right')
    plt.suptitle(title)
    fig.align_ylabels()
    plt.tight_layout()
    return fig


def plot_hmm_data_whole_session_with_states(predicted_emissions, true_emissions, predicted_states, config, model_label=None, xlim=None, xlim_orig=None, y_labels=None, title=None, savefig=False, fig_path=None, display=True):
    """Plot emissions vs. time"""
    # print("predicted_emissions, true_emissions, predicted_states", predicted_emissions, true_emissions, predicted_states)
    # print("predicted_emissions, true_emissions", np.sum(predicted_emissions), np.sum(true_emissions))
    emission_dim = predicted_emissions.shape[-1]
    fig, axs = plt.subplots(emission_dim, 1, figsize=(10, 7), sharex=True)

    xlim_ = np.r_[xlim[0]:xlim[1]+1].astype(int)
    # xlim_orig_ = np.r_[xlim_orig[0]:xlim_orig[1]].astype(int)

    d = 0
    for _ in y_labels:
        ax = axs[d] if emission_dim > 1 else axs
        max_value = max(abs(predicted_emissions[xlim_, d]).max(), abs(true_emissions[xlim_, d]).max())
        lim = 1.05 * max_value
        ax.imshow(predicted_states[xlim_][None, :], aspect="auto", interpolation="none", cmap=CMAP, vmin=0, vmax=len(COLORS)-1,
                  extent=(xlim_[0], xlim_[-1], -lim, lim), alpha=0.7)
        ax.plot(xlim_, true_emissions[xlim_, d], "k-", alpha=0.6, label='Data')
        ax.plot(xlim_, predicted_emissions[xlim_, d], 'm-', linewidth=2, label=f'{model_label}')
        ax.set_ylabel(y_labels[_], c=EC)
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

    plt.xlabel("Time (s)")
    ax = axs[0] if emission_dim > 1 else axs
    ax.legend(loc='upper right')
    plt.suptitle(title)
    fig.align_ylabels()
    plt.tight_layout()

    if savefig: plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_hmm_data_whole_session_with_states_on_top(predicted_emissions, true_emissions, predicted_states, config, model_label=None, xlim=None, xlim_orig=None, y_labels=None, title=None, savefig=False, fig_path=None, display=True):

    emission_dim = predicted_emissions.shape[-1]
    fig, axs = plt.subplots(emission_dim+1, 1, figsize=(12, 1.65*emission_dim+0.1), sharex=True, gridspec_kw={'height_ratios': [0.25] + [1]*emission_dim })

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
        ax.set_ylabel(y_labels[_], c=EC)
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


def plot_filters_statewise(orig_weights, data_config, input_labels_list, aux_input_labels_list, input_labels, y_labels, y_aux_labels, prefix, axtitles=True, skip_states=[], only_plot_inputs=None, savefig=False, fig_dir=None, display=True):
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
    fig, axs = plt.subplots(emission_dim, eff_num_states, figsize=(2.5 * eff_num_states + 3, 2.4 * len(y_labels)), sharey='row')

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
    if savefig: fig.savefig(os.path.join(fig_dir, f'{prefix}_filters_statewise_skip_states={skip_states}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
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

    print(orig_weights.shape)
    weights = basis_invtransform_one_by_one(orig_weights, basis, n_inputs)     # shape
    print(weights.shape)

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
        weights_d = orig_weights[:, [d]][..., e_mask == 1]
        print("weights_d", weights_d.shape, "n_inputs_d", n_inputs_d)
        weights_d = basis_invtransform_one_by_one(weights_d, basis, n_inputs=n_inputs_d)[:, 0]
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
    print("orig_weights.shape", orig_weights.shape)

    basis = data_config['basis']
    for z_ in range(num_states):
        print("State", z_+1)

        n_inputs_d = len(input_list)
        weights_d = orig_weights[z_][..., input_mask_by_statetrans == 1][:, None, :]
        print("weights_d", weights_d.shape, "n_inputs_d", n_inputs_d)
        weights_d = basis_invtransform_one_by_one(weights_d, basis, n_inputs=n_inputs_d)[:, 0]
        fig, axs = plt.subplots(1, n_inputs_d, figsize=(3 * n_inputs_d, 3), sharey=sharey)

        stim = 0
        for __ in input_list:
            ax = axs[stim]

            for z in range(num_states):
                if z == z_:
                    continue
                ax.plot(np.arange(-weights_d[z, stim].shape[-1], 0)/data_config['orig_fps'], weights_d[z, stim], color=COLORS[z], linewidth=3, label=f'To State {z+1}')

            ax.set_title(input_labels[__])
            ax.axhline(0, ls=':', c='k', lw=0.5)
            maxtime = weights_d[0, stim].shape[-1]//data_config['orig_fps']
            ax.set_xticks(np.linspace(-maxtime, 0, num=maxtime+1), labels=[-maxtime] + ['']*(maxtime-1) + [0])
            ax.margins(y=0.05)
            if saveindividual:
                ax.yaxis.set_tick_params(labelleft=True)  # ensure ytick labels are shown
            if ax.get_subplotspec().is_last_row() or saveindividual:
                ax.set_xlabel('Time (s)')
            if ax.get_subplotspec().is_last_col():
                ax.legend(loc='upper right')
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel('filter amplitude')
            stim += 1

        fig.supylabel(f'State transition filters\nFROM State {z_+1}', ha='center')
        plt.margins(0.05)
        plt.tight_layout()
        fig.canvas.draw()

        if savefig:
            fig.savefig(os.path.join(fig_dir, f'{filesuffix}_filters_sharey={sharey}_separate_state{z_+1}.pdf'), bbox_inches='tight', dpi=300, transparent=True)

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

    print(weights.shape)

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
        print("weights_d", weights_d.shape)
        weights_d = basis_invtransform_one_by_one(weights_d, basis, n_inputs=len(y_labels[_]))[:, 0]
        print("weights_d", weights_d.shape)
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
        if f_name in ['fFV', 'mFV']:
            t = dt
            xlim = (-1.5, 4.5)
            ylim = (0, 2)
        elif f_name in ['fLV', 'mLV']:
            t = dt
            xlim = (-2.5, 2.5)
            ylim = (0, 2)
        elif f_name in ['fAV', 'mAV']:
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

    len_eff_a_labels = len(a_labels) - len(exclude_a)
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


def plot_state_aux_dists_hist(aux_z, a_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, ax = plt.subplots(1, len(a_labels), figsize=(17, 4))

    for a, al in enumerate(a_labels):
        x99 = 0
        x0 = 0
        for z in list(aux_z.keys()):
            data = np.random.choice(np.round(aux_z[z][:, a], decimals=3), min(10000, len(aux_z[z])), replace=False)
            x0 = min(x0, 2*np.percentile(data, 0.1))
            x99 = max(x99, np.percentile(data, 99.5))
            # print(al, x0, x99)
            sns.histplot(data, ax=ax[a],
                         color=COLORS[z],
                        stat='probability',
                        label=f'State {z+1}',
                         # binwidth=1,
                        alpha=1,
                        # cut=1,
                        linewidth=2,
                        #  kde=True,
                        #  kde_kws={'cut': 0},
                        #  line_kws={'linewidth': 2},
                         bins=500,
                         element='poly',
                         fill=False,
                         # color="white",
                         # edgecolor="white"
                        # clip=(x0, x99),
                        )
        ax[a].set_xlabel(a_labels[al])
        # ax[a].set_yscale('log')
        # ax[a].margins(y=0.1)
        ax[a].set_xlim([x0-0.1, x99+0.1])
        # ax[a].axhline(0, ls=":", lw=2, c='k')
    ax[-1].legend(loc='upper right')
    fig.suptitle(f'Sensory inputs by state [{title}]')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_mean_aux_dists_hist.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_zscored_mean_aux(aux_z, a_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(3, len(aux_z.keys()), figsize=(15, 12))

    for z in list(aux_z.keys()):
        aux_means = np.mean(aux_z[z], axis=0)
        # print(aux_means, aux_means.shape)

        ax = axes[0, z] if len(aux_z.keys()) > 1 else axes[0]
        # values = aux_means
        # print("a_labels", a_labels)
        for a, al in enumerate(a_labels):
            sns.violinplot(x=a, y=aux_z[z][:, a], ax=ax, color=COLORS[z])
        # ax.bar(range(len(values)), values, color=COLORS[z])
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        # ax.set_xticks(range(len(values)))
        ax.set_xticklabels(np.array(list(a_labels.values())), rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        ax.set_title(f'State {z+1}', color=COLORS[z])
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{tick}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

        ax = axes[1, z] if len(aux_z.keys()) > 1 else axes[1]
        values = aux_means
        ax.bar(range(len(values)), values, color=COLORS[z])
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(np.array(list(a_labels.values())), rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        ax.set_title(f'State {z+1}', color=COLORS[z])
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{tick}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

        ax = axes[2, z] if len(aux_z.keys()) > 1 else axes[2]
        sorted_by = np.argsort(np.abs(aux_means))[::-1]
        ax.bar(range(len(values)), values[sorted_by], color=COLORS[z])
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(np.array(list(a_labels.values()))[sorted_by], rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        ax.set_title(f'State {z+1}', color=COLORS[z])
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)  # keeps all tick lines
        ax.set_yticklabels([f"{tick}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_zscored_mean_aux.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_aux_o_mean(aux_z, emissions_z, a_labels, o_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(1, len(aux_z.keys())*2, figsize=(25, 5), gridspec_kw={'width_ratios': [2, 1]*len(aux_z.keys())})

    for z in list(aux_z.keys()):
        aux_means = np.mean(aux_z[z], axis=0)
        o_means = np.mean(emissions_z[z], axis=0)
        print("aux", z, aux_means.shape, np.mean(aux_z[z], axis=0), np.median(aux_z[z], axis=0))
        print("out", z, o_means.shape, np.mean(emissions_z[z], axis=0), np.median(emissions_z[z], axis=0))
        print(emissions_z[z].shape)
        ax_col1, ax_col2 = 2*z, 2*z+1

        ax = axes[ax_col1]
        for a, al in enumerate(a_labels):
            print(a, al, aux_z[z][:, a].shape)
            data = aux_z[z][:, a]
            min_x, max_x = np.percentile(data, 2), np.percentile(data, 98)
            data_filtered = data[(data >= min_x) & (data <= max_x)]
            data_filtered = np.random.choice(data_filtered, size=min(100000, len(data_filtered)), replace=False)
            sns.violinplot(x=a, y=data_filtered, ax=ax, color=COLORS[z],
                           fill=False,
                           inner='quartile',
                           cut=0,  # do not extend beyond min/max of data
                           density_norm='area',  # makes violins comparable
                           # common_norm=True,
                           )
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        ax.set_xticklabels(np.array(list(a_labels.values())), rotation=90)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        ax.set_title(f'State {z+1}', color=COLORS[z], loc='right')
        # yticks = ax.get_yticks()
        # ax.set_yticks(yticks)
        # ax.set_yticklabels([f"{tick}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

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
                           # linewidth=0  # remove outline
                           # common_norm=True,
                           )
            # sns.kdeplot(y=emissions_z[z][:, o], ax=ax, color=COLORS[z])
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        ax.set_xticklabels(np.array(list(o_labels.values())), rotation=90, color=EC)
        # ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        # yticks = ax.get_yticks()
        # ax.set_yticks(yticks)
        # ax.set_yticklabels([f"{tick}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

        # ax = axes[1, ax_col1]
        # values = aux_means
        # ax.bar(range(len(values)), values, color=COLORS[z])
        # ax.axhline(0, c='k', linewidth=0.8, ls=':')
        # ax.set_xticks(range(len(values)))
        # ax.set_xticklabels(np.array(list(a_labels.values())), rotation=90)
        # if ax.get_subplotspec().is_first_col():
        #     ax.set_ylabel("z-scored value")
        # ax.margins(0.1)
        # ax.set_title(f'State {z+1}', color=COLORS[z], loc='right')
        # # yticks = ax.get_yticks()
        # # ax.set_yticks(yticks)
        # # ax.set_yticklabels([f"{tick}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick
        #
        # ax = axes[1, ax_col2]
        # values = o_means
        # print("values o_means", o_means)
        # ax.bar(range(len(values)), values, color=COLORS[z])
        # ax.axhline(0, c='k', linewidth=0.8, ls=':')
        # ax.set_xticks(range(len(values)))
        # ax.set_xticklabels(np.array(list(o_labels.values())), rotation=90, color=EC)
        # # ax.set_ylabel("z-scored value")
        # ax.margins(0.1)
        # # yticks = ax.get_yticks()
        # # ax.set_yticks(yticks)
        # # ax.set_yticklabels([f"{tick}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_zscored_mean_aux_odists.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_aux_sorted_mean_o(aux_z, emissions_z, a_labels, o_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(1, len(aux_z.keys())*2, figsize=(25, 10), gridspec_kw={'width_ratios': [2, 1]*len(aux_z.keys())})

    max_y = 0

    for z in list(aux_z.keys()):
        aux_means = np.mean(aux_z[z], axis=0)
        o_means = np.mean(emissions_z[z], axis=0)
        print("aux", z, aux_means.shape, np.mean(aux_z[z], axis=0), np.median(aux_z[z], axis=0))
        print("out", z, o_means.shape, np.mean(emissions_z[z], axis=0), np.median(emissions_z[z], axis=0))
        print(emissions_z[z].shape)

        max_y = np.max([max_y, np.max([*np.abs(aux_means), *np.abs(o_means)]) + 0.1])
        print("max_y", max_y)

        ax_col1, ax_col2 = 2*z, 2*z+1

        ax = axes[ax_col1]
        v = aux_means
        sorted_by = np.argsort(np.abs(v))[::-1]
        values = v[sorted_by]
        ax.plot([-1, len(values)], [0, 0], color='k', lw=1)
        ax.bar(range(len(values)), values, color=COLORS[z], alpha=0.6)
        # ax.axhline(0, c='r', linewidth=2, ls='-')
        # ax.set_xticks(range(len(values)))
        # ax.set_xticklabels(np.array(list(a_labels.values()))[sorted_by], rotation=90)
        # ax.set_ylabel("Relative z-scores")
        ax.margins(0.1)
        ax.set_title(f'State {z+1}', color=COLORS[z], loc='right')
        ax.axis('off')
        arrow_max = 3*max_y/4
        ax.plot([-1, -1], [-arrow_max, arrow_max], color='k', lw=3)
        ax.add_patch(FancyArrowPatch((-1, arrow_max), (-1, arrow_max+0.05), arrowstyle="->", mutation_scale=35, linewidth=3, color='k'))
        ax.add_patch(FancyArrowPatch((-1, -arrow_max), (-1, -arrow_max-0.05), arrowstyle="->", mutation_scale=35, linewidth=3, color='k'))
        for x, label in zip(range(len(values)), np.array(list(a_labels.values()))[sorted_by]):
            if np.sign(values[x]) >= 0:
                ax.text(x, 0.01, label, ha='center', va='bottom', rotation=90)
            else:
                ax.text(x, -0.01, label, ha='center', va='top', rotation=90)
        ax.text(-1.7, arrow_max-0.05, 'More', ha='center', va='bottom', rotation=90)
        ax.text(-1.7, -arrow_max+0.05, 'Less', ha='center', va='bottom', rotation=90)
        ax.text(-1.7, 0, 'Relative z-scores', ha='center', va='center', rotation=90)
        ax.set_ylim(-max_y-0.05, max_y+0.05)

        ax = axes[ax_col2]
        values = o_means
        print("values o_means", o_means)
        ax.plot([-1, len(values)], [0, 0], color='k', lw=1)
        ax.bar(range(len(values)), values, color=EC, alpha=0.7)
        # ax.axhline(0, c='k', linewidth=0.8, ls='-')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(np.array(list(o_labels.keys())), rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        ax.axis('off')
        ax.set_ylim(-max_y-0.05, max_y+0.05)
        for x, label in zip(range(len(values)), np.array(list(o_labels.values()))):
            if np.sign(values[x]) >= 0:
                ax.text(x, 0.01, label, ha='center', va='bottom', rotation=90)
            else:
                ax.text(x, -0.01, label, ha='center', va='top', rotation=90)

    # plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_zscored_mean_aux_odists_sorted.pdf'),
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


def plot_state_aux_sorted_o_mean_directional(aux_z, emissions_z, a_labels_, o_labels_, directional_vars, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(1, len(aux_z.keys())*2, figsize=(25, 10), gridspec_kw={'width_ratios': [2, 1]*len(aux_z.keys())})

    max_y = 0
    a_labels = a_labels_.copy()
    o_labels = o_labels_.copy()

    for z in list(aux_z.keys()):
        aux_means = np.zeros(len(a_labels))
        for a, al in enumerate(a_labels):
            # print(a, al, aux_means[a])
            if al in directional_vars:
                aux_means[a] = np.mean(np.abs(aux_z[z][:, a]))
                a_labels[al] = directional_vars[al]
            else:
                aux_means[a] = np.mean(aux_z[z][:, a])

        o_means = np.zeros(len(o_labels))
        for o, ol in enumerate(o_labels):
            if ol in directional_vars:
                o_means[o] = np.mean(np.abs(emissions_z[z][:, o]))
                o_labels[ol] = directional_vars[ol]
            else:
                o_means[o] = np.mean(emissions_z[z][:, o])

        print("aux", z, aux_means.shape, np.mean(aux_z[z], axis=0), np.median(aux_z[z], axis=0))
        print("out", z, o_means.shape, np.mean(emissions_z[z], axis=0), np.median(emissions_z[z], axis=0))
        print(emissions_z[z].shape)

        max_y = np.max([max_y, np.max([*np.abs(aux_means), *np.abs(o_means)]) + 0.1])
        print("max_y", max_y)

        ax_col1, ax_col2 = 2*z, 2*z+1

        ax = axes[ax_col1]
        v = aux_means
        sorted_by = np.argsort(np.abs(v))[::-1]
        values = v[sorted_by]
        ax.plot([-1, len(values)], [0, 0], color='k', lw=1)
        ax.bar(range(len(values)), values, color=COLORS[z], alpha=0.6)
        # ax.axhline(0, c='r', linewidth=2, ls='-')
        # ax.set_xticks(range(len(values)))
        # ax.set_xticklabels(np.array(list(a_labels.values()))[sorted_by], rotation=90)
        # ax.set_ylabel("Relative z-scores")
        ax.margins(0.1)
        ax.set_title(f'State {z+1}', color=COLORS[z], loc='right')
        ax.axis('off')
        arrow_max = 3*max_y/4
        ax.plot([-1, -1], [-arrow_max, arrow_max], color='k', lw=3)
        ax.add_patch(FancyArrowPatch((-1, arrow_max), (-1, arrow_max+0.05), arrowstyle="->", mutation_scale=35, linewidth=3, color='k'))
        ax.add_patch(FancyArrowPatch((-1, -arrow_max), (-1, -arrow_max-0.05), arrowstyle="->", mutation_scale=35, linewidth=3, color='k'))
        for x, label in zip(range(len(values)), np.array(list(a_labels.values()))[sorted_by]):
            if np.sign(values[x]) >= 0:
                ax.text(x, 0.01, label, ha='center', va='bottom', rotation=90)
            else:
                ax.text(x, -0.01, label, ha='center', va='top', rotation=90)
        ax.text(-1.7, arrow_max-0.05, 'More', ha='center', va='bottom', rotation=90)
        ax.text(-1.7, -arrow_max+0.05, 'Less', ha='center', va='bottom', rotation=90)
        ax.text(-1.7, 0, 'Relative z-scores', ha='center', va='center', rotation=90)
        ax.set_ylim(-max_y-0.05, max_y+0.05)

        ax = axes[ax_col2]
        values = o_means
        print("values o_means", o_means)
        ax.plot([-1, len(values)], [0, 0], color='k', lw=1)
        ax.bar(range(len(values)), values, color=EC, alpha=0.7)
        # ax.axhline(0, c='k', linewidth=0.8, ls='-')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(np.array(list(o_labels.keys())), rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        ax.axis('off')
        ax.set_ylim(-max_y-0.05, max_y+0.05)
        for x, label in zip(range(len(values)), np.array(list(o_labels.values()))):
            if np.sign(values[x]) >= 0:
                ax.text(x, 0.01, label, ha='center', va='bottom', rotation=90)
            else:
                ax.text(x, -0.01, label, ha='center', va='top', rotation=90)

    # plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_zscored_mean_aux_odists_sorted_abs_directional.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_dwell_times(dwell_times_z, num_states, effective_fps, title='', savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # for z, durations in dwell_times_z.items():
    #     print(f"State {z+1}: Mean dwell time = {np.mean(durations):.2f}, n = {len(durations)}")

    durations = []
    for z in range(num_states):
        d = dwell_times_z[z] / effective_fps    # in seconds
        # print(f"State {z + 1}: Mean dwell time = {np.mean(dwell_times_z[z]):.2f}, n = {len(dwell_times_z[z])}")
        print(f"State {z + 1}: Mean dwell time = {np.mean(d):.2f}s")
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
        # bw_adjust=0.5,
        cut=0,
        # clip=(0, None),
        # common_norm=True,
        linewidth=3,
        legend=None,
    )
    x = list(range(0, 4))
    plt.xlim([x[0], x[-1]])
    plt.xticks(x)
    plt.xlabel("Time (s)")
    plt.title("State residency")

    # ax.get_legend()
    # ax.legend()
    # legend.set_title(None)
    # legend.set_loc('upper right')

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'state_dwell_times.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_dwell_times_gkde(dwell_times_z, num_states, effective_fps, title='', savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(6, 4))

    from scipy.stats import gaussian_kde
    for z, durations in dwell_times_z.items():
        durations = durations / effective_fps
        if len(durations) > 1:  # KDE needs >1 data point
            kde = gaussian_kde(durations)
            x = np.arange(-2, max(durations) + 1)
            pdf_vals = kde(x)
            pdf_vals /= pdf_vals.sum()  # normalize to make it sum to 1
            plt.plot(x, pdf_vals, label=f"State {z+1}", c=COLORS[z], linewidth=2)
    x = list(range(0, 4))
    plt.xlim([x[0], x[-1]])
    plt.xticks(x)
    plt.ylabel("p(state)")
    plt.xlabel("Time (s)")
    plt.title("State residency")
    plt.legend(loc='upper right')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'state_dwell_times_gkde.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_prob_states(state_seqs, config, title=None, savefig=False, fig_dir=None, display=True):
    # print("state_seqs", state_seqs.shape)

    fig = plt.figure(figsize=(10, 5))
    for z in range(config['num_states']):
        print(state_seqs)
        prob_z = np.mean(state_seqs == z, axis=0)  # Probability of z at each time step
        plt.plot(uniform_filter1d(prob_z, size=100), c=COLORS[z], linewidth=1.5, label=f'State {z+1}')
    plt.xlabel('Time', fontsize='large')
    plt.legend(loc='upper right')
    plt.margins(0.05)
    plt.ylabel('P(state)', fontsize='large')
    # plt.xticks([0, len(prob_z)], [0, 15], fontsize='medium')
    plt.yticks(fontsize='medium')
    plt.title(f'State occupancy')
    plt.ylim(0, 1)
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_prob_states_over_time.pdf'), bbox_inches='tight', dpi=300,
                            transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_prob_states_aligned(resampled_state_seq, n_le=None, uniform_filter_size=1, config=None, title=None, xticks=None, xlabel=None, savefig=False, fig_dir=None, display=True):

    fig = plt.figure(figsize=(9, 3))

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
    plt.ylim(0, 0.7)
    plt.xlabel(xlabel)
    plt.ylabel('p(state)')
    # plt.legend(loc='upper left')
    # if title: plt.title(f'{title} sessions')
    if xticks: plt.xticks([0, GRID-1], xticks)
    plt.yticks(ticks=np.linspace(0, 0.7, num=8), labels=[0] + ['']*6 + [0.7])
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
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
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


def plot_ethogram_community(transition_matrix, threshold, savefig=False, fig_dir=None, display=True):

    fig = plt.figure(figsize=(10, 10))

    # Define positions using a circular layout
    G = nx.DiGraph()
    num_states = transition_matrix.shape[0]

    # Add edges with weights
    for i in range(num_states):
        for j in range(num_states):
            if transition_matrix[i, j] > threshold:  # Only add edges with nonzero probability
                G.add_edge(i, j, weight=transition_matrix[i, j])

    communities = nx.community.greedy_modularity_communities(G)
    # print("communities", communities)

    # Compute positions for the node clusters as if they were themselves nodes in a
    # supergraph using a larger scale factor
    supergraph = nx.cycle_graph(len(communities))
    superpos = nx.spring_layout(G, scale=50)

    # Use the "supernode" positions as the center of each node cluster
    centers = list(superpos.values())
    pos = {}
    for center, comm in zip(centers, communities):
        pos.update(nx.spring_layout(nx.subgraph(G, comm), center=center))

    edges = G.edges(data=True)
    edge_widths = [d['weight'] * 5 for (u, v, d) in edges]  # Differentiate in/out degrees

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000,
            font_size=14, font_weight='bold', edge_color='gray', width=edge_widths,
            arrows=True,
            connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edges(G, pos=pos)

    # Draw edge labels
    edge_labels = {(u, v): f"{d['weight']:.3f}" for (u, v, d) in edges}
    # print(edge_labels, len(edge_labels))
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.4, connectionstyle='arc3,rad=0.1')

    plt.title("Transition Probability Graph")
    if savefig: fig.savefig(os.path.join(fig_dir, f'ethogram_community_{threshold}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
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


def plot_empirical_occupancy(state_seqs, config, title=None, savefig=False, fig_dir=None, display=True):
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


def plot_var_explained(train_r2, test_r2, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot overall r2 scores.
    :return:
    """
    fig = plt.figure(figsize=(3, 4))
    plt.plot(1, train_r2 * 100, 'ko', mfc='none', label='Train', markersize=10)
    plt.plot(1, test_r2 * 100, 'ko', label='Held-out', markersize=10)
    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks([])
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    plt.legend(loc='lower right')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'overall_r2_scores.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return fig


def plot_pearson(train_pr, test_pr, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot pearsonr scores.
    :return:
    """
    fig = plt.figure(figsize=(3, 4))
    plt.plot(1, train_pr, 'ko', mfc='none', label='Train', markersize=10)
    plt.plot(1, test_pr, 'ko', label='Held-out', markersize=10)
    plt.ylabel(r"Pearson $r$")
    plt.margins(0.1)
    plt.xticks([])
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    plt.legend(loc='lower right')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'overall_pearson_scores.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return fig


def plot_ll(train_lp, test_lp, config, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot overall LL scores.
    :return:
    """
    fig = plt.figure(figsize=(3, 4))
    train_lp = train_lp * config['effective_fps']/np.log(2)
    test_lp = test_lp * config['effective_fps']/np.log(2)

    plt.plot(1, train_lp, 'ko', mfc='none', label='Train', markersize=10)
    plt.plot(1, test_lp, 'ko', label='Held-out', markersize=10)
    plt.ylabel('Normalized LL\n(bits/s)')
    plt.margins(0.1)
    plt.xticks([])
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    plt.legend(loc='lower right')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'overall_ll_scores.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return fig


def plot_var_explained_by_fly(train_r2s, test_r2s, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot r2 scores, by fly.
    :return:
    """
    fig = plt.figure(figsize=(4, 5))

    train_jitter = np.random.uniform(-0.1, 0.1, len(train_r2s))
    test_jitter = np.random.uniform(-0.1, 0.1, len(test_r2s))

    plt.plot(train_jitter+1, train_r2s * 100, 'ko', mfc='none', label='Train', markersize=7)
    plt.errorbar(1.2, np.mean(train_r2s * 100), yerr=np.std(train_r2s * 100), color='k', fmt='o', capsize=0)

    plt.plot(test_jitter+2, test_r2s * 100, 'ko', label='Held-out', markersize=7)
    plt.errorbar(2.2, np.mean(test_r2s * 100), yerr=np.std(test_r2s * 100), color='k', fmt='o', capsize=0)

    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks([])
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    plt.legend(loc='lower right')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'r2_scores_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return fig


def plot_pearson_by_fly(train_prs, test_prs, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot pearsonr scores, by fly.
    :return:
    """
    fig = plt.figure(figsize=(4, 5))

    train_jitter = np.random.uniform(-0.1, 0.1, len(train_prs))
    test_jitter = np.random.uniform(-0.1, 0.1, len(test_prs))

    plt.plot(train_jitter+1, train_prs, 'ko', mfc='none', label='Train', markersize=7)
    plt.errorbar(1.2, np.mean(train_prs), yerr=np.std(train_prs), color='k', fmt='o', capsize=0)

    plt.plot(test_jitter+2, test_prs, 'ko', label='Held-out', markersize=7)
    plt.errorbar(2.2, np.mean(test_prs), yerr=np.std(test_prs), color='k', fmt='o', capsize=0)

    plt.ylabel(r"Pearson $r$")
    plt.margins(0.1)
    plt.xticks([])
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    plt.legend(loc='lower right')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'pearson_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return fig


def plot_ll_by_fly(train_lls, test_lls, config, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot ll scores, by fly.
    :return:
    """
    fig = plt.figure(figsize=(3, 4))

    train_jitter = np.random.uniform(-0.1, 0.1, len(train_lls))
    test_jitter = np.random.uniform(-0.1, 0.1, len(test_lls))

    train_lls = train_lls * config['effective_fps']/np.log(2)
    test_lls = test_lls * config['effective_fps']/np.log(2)

    plt.plot(train_jitter+1, train_lls, 'ko', mfc='none', label='Train', markersize=7)
    plt.errorbar(1.2, np.mean(train_lls), yerr=np.std(train_lls), color='k', fmt='o', capsize=0)

    plt.plot(test_jitter+2, test_lls, 'ko', label='Held-out', markersize=7)
    plt.errorbar(2.2, np.mean(test_lls), yerr=np.std(test_lls), color='k', fmt='o', capsize=0)

    plt.ylabel('Normalized LL\n(bits/s)')
    plt.margins(0.1)
    plt.xticks([])
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    plt.legend(loc='lower right')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'll_scores_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return fig


def plot_var_explained_ind(r2_scores, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot overall r2 scores.
    :return:
    """
    fig = plt.figure(figsize=(3, 4))
    r2_scores = np.where(np.abs(r2_scores) > 1e2, 0, r2_scores)
    print(r2_scores)
    plt.plot(np.ones(r2_scores.size) + np.random.uniform(-0.1, 0.1, size=r2_scores.size), r2_scores, 'b.', markersize=7)
    plt.boxplot(r2_scores, positions=[1.2], widths=0.1)
    # plt.scatter([1] * len(r2_scores), r2_scores, 'b.')
    # plt.plot(r2_scores)
    plt.ylabel('Var explained (%)')
    plt.margins(0.3)
    plt.xticks([0, 1, 2], labels=[])
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    # plt.legend(loc='lower left')
    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'overall_ind_r2_scores.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_var_explained_by_z(r2_z, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot overall r2 scores in each state.
    :return:
    """
    fig = plt.figure()
    for z in r2_z:
        plt.bar(z+1, r2_z[z] * 100, color=COLORS[z])
    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks(np.array(list(r2_z.keys())).astype(int) + 1)
    plt.xlabel('State')
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    # plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_z.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_pearson_by_z(pearson_z, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot pearsonr scores in each state.
    :return:
    """
    fig = plt.figure()
    for z in pearson_z:
        plt.bar(z+1, pearson_z[z], color=COLORS[z])
    plt.ylabel('Pearson correlation coefficient')
    plt.margins(0.1)
    plt.xticks(np.array(list(pearson_z.keys())).astype(int) + 1)
    plt.xlabel('State')
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    # plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_pearson_by_z.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_pearson_by_z_vs_all(pearson_all, pearson_z, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot pearsonr scores in each state vs all
    :return:
    """
    fig = plt.figure()
    plt.bar(0, pearson_all, color='gray')
    for z in pearson_z:
        plt.bar(z+1, pearson_z[z], color=COLORS[z])
    plt.ylabel('Pearson correlation coefficient')
    plt.margins(0.1)
    # plt.xticks(np.array(list(pearson_z.keys())).astype(int) + 1)
    plt.xticks([0] + [_+1 for _ in list(pearson_z.keys())], ['All'] + [_+1 for _ in list(pearson_z.keys())])
    plt.xlabel('State')
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    # plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_pearson_by_z_vs_all.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_var_explained_by_z_by_fly(r2_z, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot overall r2 scores in each state, by fly.
    :return:
    """
    fig = plt.figure()
    sort_by = np.argsort(r2_z[0])[::-1]
    colors = np.linspace(0, 1, len(r2_z[0]))
    n_colors = len(r2_z[0])

    for z in r2_z:
        # cmap = LinearSegmentedColormap.from_list("xkcd_reverse_fade", [COLORS[z], "#ffffff"])

        base_rgba = to_rgba(COLORS[z], alpha=1.0)
        faded_colors = np.tile(base_rgba, (n_colors, 1))
        faded_colors[:, -1] = np.linspace(1, 0.3, n_colors)
        transparent_cmap = ListedColormap(faded_colors)

        jitter = np.random.uniform(-0.1, 0.1, len(r2_z[z])) + 1
        plt.scatter(z + jitter, r2_z[z][sort_by] * 100, c=colors, cmap=transparent_cmap, s=SCATTERSIZE)
        plt.errorbar(z + 1.2, np.mean(r2_z[z] * 100), yerr=np.std(r2_z[z] * 100), color='k', alpha=0.5, fmt='o', capsize=0)

    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks(np.array(list(r2_z.keys())).astype(int) + 1)
    plt.xlabel('State')
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    # plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_z_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_pearson_by_z_by_fly(pearson_z, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot pearsonr scores in each state, by fly.
    :return:
    """
    fig = plt.figure()
    sort_by = np.argsort(pearson_z[0])[::-1]
    colors = np.linspace(0, 1, len(pearson_z[0]))
    n_colors = len(pearson_z[0])

    for z in pearson_z:
        # cmap = LinearSegmentedColormap.from_list("xkcd_reverse_fade", [COLORS[z], "#ffffff"])

        base_rgba = to_rgba(COLORS[z], alpha=1.0)
        faded_colors = np.tile(base_rgba, (n_colors, 1))
        faded_colors[:, -1] = np.linspace(1, 0.3, n_colors)
        transparent_cmap = ListedColormap(faded_colors)

        jitter = np.random.uniform(-0.1, 0.1, len(pearson_z[z])) + 1
        plt.scatter(z + jitter, pearson_z[z][sort_by], c=colors, cmap=transparent_cmap, s=SCATTERSIZE, edgecolors='none')
        plt.errorbar(z + 1.2, np.mean(pearson_z[z]), yerr=np.std(pearson_z[z]), color='k', alpha=0.5, fmt='o', capsize=0)

    plt.ylabel(r"Pearson $r$")
    plt.margins(0.1)
    plt.xticks(np.array(list(pearson_z.keys())).astype(int) + 1)
    plt.xlabel('State')
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    # plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_pearson_by_z_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_pearson_by_z_by_fly_vs_all(pearson_all, pearson_z, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot pearsonr scores in each state, by fly vs all states together
    :return:
    """
    fig = plt.figure(figsize=(5, 5))
    sort_by = np.argsort(pearson_z[0])[::-1]
    colors = np.linspace(0, 1, len(pearson_z[0]))
    n_colors = len(pearson_z[0])

    base_rgba = to_rgba('gray', alpha=1.0)
    faded_colors = np.tile(base_rgba, (n_colors, 1))
    faded_colors[:, -1] = np.linspace(1, 0.3, n_colors)
    transparent_cmap = ListedColormap(faded_colors)
    jitter = np.random.uniform(-0.1, 0.1, len(pearson_all))
    plt.scatter(0 + jitter, pearson_all[sort_by], c=colors, cmap=transparent_cmap, s=SCATTERSIZE, edgecolors='none')
    plt.errorbar(0 + .2, np.mean(pearson_all), yerr=np.std(pearson_all), color='k', alpha=0.5, fmt='o', capsize=0)

    for z in pearson_z:
        # cmap = LinearSegmentedColormap.from_list("xkcd_reverse_fade", [COLORS[z], "#ffffff"])

        base_rgba = to_rgba(COLORS[z], alpha=1.0)
        faded_colors = np.tile(base_rgba, (n_colors, 1))
        faded_colors[:, -1] = np.linspace(1, 0.3, n_colors)
        transparent_cmap = ListedColormap(faded_colors)

        jitter = np.random.uniform(-0.1, 0.1, len(pearson_z[z]))
        plt.scatter(z+1 + jitter, pearson_z[z][sort_by], c=colors, cmap=transparent_cmap, s=SCATTERSIZE, edgecolors='none')
        plt.errorbar(z+1 + .2, np.mean(pearson_z[z]), yerr=np.std(pearson_z[z]), color='k', alpha=0.5, fmt='o', capsize=0)

    plt.ylabel(r"Pearson $r$")
    plt.margins(0.1)
    # plt.xticks(np.array(list(pearson_z.keys())).astype(int) + 1)
    plt.xticks([0] + [_ + 1 for _ in list(pearson_z.keys())], ['All'] + [_ + 1 for _ in list(pearson_z.keys())])
    plt.xlabel('State')
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    # plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_pearson_by_z_by_fly_vs_all.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_var_explained_by_o(r2_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot overall r2 scores for each emission dim.
    :return:
    """
    fig = plt.figure()
    for o in r2_o:
        plt.bar(o, r2_o[o]*100, color=EC, width=0.6)
    plt.xticks(list(r2_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks(list(r2_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return fig


def plot_var_explained_by_o_by_fly(r2_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot overall r2 scores for each emission dim, by fly.
    :return:
    """
    fig = plt.figure()
    sort_by = np.argsort(r2_o[0])[::-1]
    colors = np.linspace(0, 1, len(r2_o[0]))
    for o in r2_o:
        r2s = r2_o[o]
        jitter = np.random.uniform(-0.1, 0.1, len(r2s))
        plt.scatter(o + jitter, r2s[sort_by] * 100, c=colors, cmap='PRGn', s=SCATTERSIZE, edgecolors='none')
        plt.errorbar(o + 0.2, np.mean(r2s * 100), yerr=np.std(r2s * 100), color='k', alpha=0.5, fmt='o', capsize=0)
    plt.xticks(list(r2_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks(list(r2_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_o_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return fig


def plot_correlation_by_o(corr_z, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot correlation coefficients for each emission timeseries.
    :return:
    """
    fig = plt.figure()
    for o in corr_z:
        plt.bar(o, corr_z[o], color=EC, width=0.6)
    plt.xticks(list(corr_z.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Correlation coefficient')
    plt.margins(0.1)
    plt.xticks(list(corr_z.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_correlation_by_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_correlation_by_o_by_fly(corr_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot correlation coefficients for each emission timeseries, by fly.
    :return:
    """
    fig = plt.figure(figsize=(6, 5))
    sort_by = np.argsort(corr_o[0])[::-1]
    colors = np.linspace(0, 1, len(corr_o[0]))  # so colors are now in order of decreasing correlation scores for emission0
    for o in corr_o:
        coors = corr_o[o]
        jitter = np.random.uniform(-0.1, 0.1, len(coors))
        plt.scatter(o + jitter, coors[sort_by], c=colors, cmap=ECmap, s=SCATTERSIZE, edgecolors='none')
        plt.errorbar(o + 0.2, np.mean(coors), yerr=np.std(coors), color='k', alpha=0.5, fmt='o', capsize=0)

    plt.xticks(list(corr_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel(r"Pearson $r$")
    plt.margins(0.1)
    plt.xticks(list(corr_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_correlation_by_o_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_correlation_lags_by_o(lags_o, o_labels, effective_fps, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot correlation max lags for each emission timeseries.
    :return:
    """
    fig = plt.figure()
    for o in lags_o:
        plt.bar(o, (lags_o[o] * 1000) / effective_fps, color=EC, width=0.3)
    plt.xticks(list(lags_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Lag for max correlation coefficient (ms)')
    plt.margins(0.1)
    plt.xticks(list(lags_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_correlation_max_lags_by_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_correlation_lags_by_o_by_fly(lags_o, o_labels, effective_fps, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot correlation max lags for each emission timeseries., by fly.
    :return:
    """
    fig = plt.figure()
    sort_by = np.argsort(lags_o[0])[::-1]
    colors = np.linspace(0, 1, len(lags_o[0]))  # so colors are now in order of decreasing lags for emission0
    for o in lags_o:
        lags = (lags_o[o] * 1000) / effective_fps
        jitter = np.random.uniform(-0.1, 0.1, len(lags))
        plt.scatter(o + jitter, lags[sort_by], c=colors, cmap='BrBG', s=SCATTERSIZE, edgecolors='none')
        plt.errorbar(o + 0.2, np.mean(lags), yerr=np.std(lags), color='k', alpha=0.5, fmt='o', capsize=0)

    plt.xticks(list(lags_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Lag for max correlation coefficient (ms)')
    plt.margins(0.1)
    plt.xticks(list(lags_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    # plt.title(title)
    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_correlation_max_lags_by_o_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return fig


def plot_var_explained_by_z_o(r2_z_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot r2 scores in each state for each emission dimension separately
    """
    figwidth = 3*len(r2_z_o) + 0.5
    fig, ax = plt.subplots(1, len(r2_z_o), figsize=(figwidth, 7), sharey=True, layout='constrained')
    for z in r2_z_o:
        axes = ax[z] if len(r2_z_o) > 1 else ax
        for o in r2_z_o[z]:
            axes.bar(o, r2_z_o[z][o] * 100, color=COLORS[z])
        axes.set_xticks(list(r2_z_o[z].keys()), list(o_labels.values()), rotation=0)
        axes.set_title(f'State {z+1}', color=COLORS[z])
        axes.axhline(0, c='k', ls=':', lw=2)
        axes.margins(0.1)

    axes = ax[0] if len(r2_z_o) > 1 else ax
    axes.set_ylabel('Var explained (%)')
    # plt.suptitle(title)
    # plt.ylim(-10, 20)
    # plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    return fig


def plot_pearson_by_z_o(pearson_z_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot pearsonr in each state for each emission dimension separately
    """
    figwidth = 3*len(pearson_z_o) + 0.5
    fig, ax = plt.subplots(1, len(pearson_z_o), figsize=(figwidth, 7), sharey=True, layout='constrained')
    for z in pearson_z_o:
        axes = ax[z] if len(pearson_z_o) > 1 else ax
        for o in pearson_z_o[z]:
            axes.bar(o, pearson_z_o[z][o], color=COLORS[z])
        axes.set_xticks(list(pearson_z_o[z].keys()), list(o_labels.values()), rotation=0)
        axes.set_title(f'State {z+1}', color=COLORS[z])
        axes.axhline(0, c='k', ls=':', lw=2)
        axes.margins(0.1)

    axes = ax[0] if len(pearson_z_o) > 1 else ax
    axes.set_ylabel('Pearson correlation coefficient (r)')
    # plt.suptitle(title)
    # plt.ylim(-10, 20)
    # plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_pearson_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    return fig


def plot_auxem_acc_by_z_o(stateseq, aux_emissions, probs_z_o, config, ay_labels, skip_states=[], title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot accuracy scores in each state for each auxiliary emission dimension separately
    """
    from sklearn.metrics import classification_report, confusion_matrix

    num_states = config['num_states']
    aux_emission_dim = len(ay_labels)
    aux_emissions_ = np.concatenate(aux_emissions, axis=0)
    z_seq_ = np.concatenate(stateseq, axis=0)

    acc_z_o = {}
    counts_z_o = {}
    for z in range(num_states):
        acc_z_o[z] = {}
        counts_z_o[z] = {}
        z_mask = (z_seq_ == z)
        for o in range(aux_emission_dim):
            y_true = aux_emissions_[z_mask][:, o]
            counts_z_o[z][o] = np.sum(y_true) / len(y_true)
            y_pred = probs_z_o[z][o] >= 0.65
            acc_z_o[z][o] = f1_score(y_true, y_pred)
            if not np.sum(y_pred):
                print(f"z {z}", classification_report(y_true, y_pred))
                print(f"z {z}", confusion_matrix(y_true, y_pred))

    print("acc_z_o", acc_z_o)
    plt.figure(figsize=(6, 5))
    xt = []
    for z in acc_z_o:
        if z in skip_states: continue
        plt.bar(z, acc_z_o[z][0], color=COLORS[z])
        xt.append(z)
    xt = np.array(xt)
    plt.ylim(0, 0.5)
    plt.ylabel('F1 score')
    plt.xlabel('State')
    plt.xticks(xt, xt+1)
    plt.tight_layout()
    if savefig: plt.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_skip_states={len(skip_states)}_auxem_acc_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()

    print("counts", counts_z_o)
    plt.figure(figsize=(6, 5))
    xt = []
    for z in counts_z_o:
        if z in skip_states: continue
        plt.bar(z, counts_z_o[z][0], color=COLORS[z])
        xt.append(z)
    xt = np.array(xt)
    plt.ylim(0, 1)
    plt.ylabel('P(wing flick = 1 | state)')
    plt.xlabel('State')
    plt.xticks(xt, xt+1)
    plt.tight_layout()
    if savefig: plt.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_skip_states={len(skip_states)}_auxem_fraction_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


# def plot_auxem_acc_by_z_o_traintest(train_stateseq, test_stateseq, train_aux_emissions, test_aux_emissions, train_predict_probs_z_o, test_predict_probs_z_o, config, ay_labels, skip_states=[], title=None, savefig=False, fig_dir=None, display=True):
#     """
#     Plot accuracy scores in each state for each auxiliary emission dimension separately
#     """
#     from sklearn.metrics import classification_report, confusion_matrix
#
#     num_states = config['num_states']
#     aux_emission_dim = len(ay_labels)
#
#     def get_acc_z_o(stateseq, aux_emissions, probs_z_o):
#         aux_emissions_ = np.concatenate(aux_emissions, axis=0)
#         z_seq_ = np.concatenate(stateseq, axis=0)
#         acc_z_o = {}
#         counts_z_o = {}
#         for z in range(num_states):
#             acc_z_o[z] = {}
#             counts_z_o[z] = {}
#             for o in range(aux_emission_dim):
#                 y_true = aux_emissions_[z_seq_ == z][:, o]
#                 counts_z_o[z][o] = np.sum(y_true) / len(y_true)
#                 y_pred = probs_z_o[z][o] >= 0.5
#                 acc_z_o[z][o] = f1_score(y_true, y_pred)
#                 if not np.sum(y_pred):
#                     print(f"z {z}", classification_report(y_true, y_pred))
#                     print(f"z {z}", confusion_matrix(y_true, y_pred))
#         return acc_z_o, counts_z_o
#
#     def get_acc(aux_emissions, probs_o):
#         aux_emissions_ = np.concatenate(aux_emissions, axis=0)
#         acc_o = {}
#         counts_o = {}
#         for o in range(aux_emission_dim):
#             y_true = aux_emissions_[:, o]
#             counts_o[o] = np.sum(y_true) / len(y_true)
#             y_pred = probs_o[o] >= 0.5
#             acc_o[o] = f1_score(y_true, y_pred)
#             if not np.sum(y_pred):
#                 print(classification_report(y_true, y_pred))
#                 print(confusion_matrix(y_true, y_pred))
#         return acc_o, counts_o
#
#     train_acc_o, train_counts_o = get_acc(train_aux_emissions, train_predict_probs_z_o)
#     test_acc_o, test_counts_o = get_acc(test_aux_emissions, test_predict_probs_z_o)
#
#     train_acc_z_o, train_counts_z_o = get_acc_z_o(train_stateseq, train_aux_emissions, train_predict_probs_z_o)
#     print("train_acc_z_o", train_acc_z_o)
#     test_acc_z_o, test_counts_z_o = get_acc_z_o(test_stateseq, test_aux_emissions, test_predict_probs_z_o)
#     print("test_acc_z_o", test_acc_z_o)
#
#     plt.figure(figsize=(5.5, 4.5))
#     xt = []
#     for z in train_acc_z_o:
#         if z in skip_states: continue
#         plt.bar(z-0.2, train_acc_z_o[z][0], color=COLORS[z], alpha=0.5, width=0.2, label='train data' if z == 0 else '')
#         plt.bar(z+0.2, test_acc_z_o[z][0], color=COLORS[z], width=0.2, label='held-out data' if z == 0 else '')
#         xt.append(z)
#     xt = np.array(xt)
#     plt.legend(loc='upper right')
#     plt.ylim(0, 0.5)
#     plt.ylabel('F1 score')
#     plt.xlabel('State')
#     plt.xticks(xt, xt+1)
#     plt.tight_layout()
#     if savefig: plt.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_skip_states={len(skip_states)}_traintest_auxem_acc_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
#     if display: plt.show()
#     plt.close()
#
#     print("train_counts_z_o", train_counts_z_o)
#     print("test_counts_z_o", test_counts_z_o)
#     plt.figure(figsize=(5.5, 4.5))
#     xt = []
#     for z in train_counts_z_o:
#         if z in skip_states: continue
#         plt.bar(z-0.2, train_counts_z_o[z][0], color=COLORS[z], alpha=0.5, width=0.2, label='train data' if z == 0 else '')
#         plt.bar(z+0.2, test_counts_z_o[z][0], color=COLORS[z], width=0.2, label='held-out data' if z == 0 else '')
#         xt.append(z)
#     xt = np.array(xt)
#     plt.ylim(0, 1)
#     plt.legend(loc='upper right')
#     plt.ylabel('P(wing flick = 1 | state)')
#     plt.xlabel('State')
#     plt.xticks(xt, xt+1)
#     plt.tight_layout()
#     if savefig: plt.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_skip_states={len(skip_states)}_traintest_auxem_fraction_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
#     if display: plt.show()
#     plt.close()
#     return


def plot_auxem_acc_full_precomputed(train_true_z_o, train_preds_z_o, test_true_z_o, test_preds_z_o, title=None, savefig=False, fig_dir=None, display=True):

    train_alldata = [np.vstack([train_true_z_o[z][0], train_preds_z_o[z][0]]).T for z in train_true_z_o]
    train_alldata = np.vstack(train_alldata)
    print(train_alldata.shape)
    train_y_true = train_alldata[:, 0]
    train_y_pred = train_alldata[:, 1]
    train_f1_score = f1_score(train_y_true, train_y_pred)

    test_alldata = [np.vstack([test_true_z_o[z][0], test_preds_z_o[z][0]]).T for z in test_true_z_o]
    test_alldata = np.vstack(test_alldata)
    print(test_alldata.shape)
    test_y_true = test_alldata[:, 0]
    test_y_pred = test_alldata[:, 1]
    test_f1_score = f1_score(test_y_true, test_y_pred)

    scores = {
        'train': train_f1_score,
        'test': test_f1_score,
    }
    for s in scores:
        plt.figure(figsize=(2, 5))
        plt.bar(0, scores[s], width=0.4, color=EC)
        plt.xticks([0], ['wing\nflick'])
        plt.ylim(0, 0.8)
        plt.ylabel('F1 score')
        plt.tight_layout()
        if savefig: plt.savefig(os.path.join(fig_dir, f'{s}_full_precomp_auxem_f1score_by_z.pdf'),
                                bbox_inches='tight', dpi=300, transparent=True)
        if display: plt.show()
        plt.close()
    return


def plot_auxem_frac_full_precomputed(train_true_z_o, test_true_z_o, title=None, savefig=False, fig_dir=None, display=True):

    train_alldata = [train_true_z_o[z][0] for z in train_true_z_o]
    train_alldata = np.concatenate(train_alldata)
    print(train_alldata.shape)
    train_frac = np.sum(train_alldata)/len(train_alldata)

    test_alldata = [test_true_z_o[z][0] for z in test_true_z_o]
    test_alldata = np.concatenate(test_alldata)
    print(test_alldata.shape)
    test_frac = np.sum(test_alldata)/len(test_alldata)

    fracs = {
        'train': train_frac,
        'test': test_frac,
    }
    for s in fracs:
        plt.figure(figsize=(2, 5))
        plt.bar(0, fracs[s], width=0.4, color=EC)
        plt.xticks([0], ['wing\nflick'])
        plt.ylim(0, 0.5)
        plt.ylabel('P(wing flick = 1 | state)')
        plt.tight_layout()
        if savefig: plt.savefig(os.path.join(fig_dir, f'{s}_full_precomp_auxem_frac.pdf'),
                                bbox_inches='tight', dpi=300, transparent=True)
        if display: plt.show()
        plt.close()
    return


def plot_auxem_acc_by_z_o_traintest_precomputed(train_f1score_z_o, test_f1score_z_o, config, ay_labels, skip_states=[], title=None, savefig=False, fig_dir=None, display=True):
    scores = {
        'train': train_f1score_z_o,
        'test': test_f1score_z_o,
    }
    for s in ['train', 'test']:
        f1score_z_o = scores[s]
        plt.figure(figsize=(3, 4.5))
        xt = []
        for z in f1score_z_o:
            if z in skip_states: continue
            plt.bar(z, f1score_z_o[z][0], color=COLORS[z], width=0.5)
            xt.append(z)
        xt = np.array(xt)
        plt.ylim(0, 0.8)
        plt.ylabel('F1 score')
        plt.xlabel('State')
        plt.xticks(xt, xt + 1)
        plt.tight_layout()
        if savefig: plt.savefig(os.path.join(fig_dir, f'{s}_skip_states={len(skip_states)}_precomp_auxem_f1score_by_z.pdf'),
                                bbox_inches='tight', dpi=300, transparent=True)
        if display: plt.show()
        plt.close()
    return


def plot_auxem_frac_by_z_o_traintest_precomputed(train_true_z_o, test_true_z_o, config, ay_labels, skip_states=[], title=None, savefig=False, fig_dir=None, display=True):
    scores = {
        'train': train_true_z_o,
        'test': test_true_z_o,
    }
    for s in ['train', 'test']:
        true_z_o = scores[s]
        plt.figure(figsize=(3, 4.5))
        xt = []
        for z in true_z_o:
            if z in skip_states: continue
            frac = np.sum(true_z_o[z][0]) / len(true_z_o[z][0])
            plt.bar(z, frac, color=COLORS[z], width=0.5)
            xt.append(z)
        xt = np.array(xt)
        plt.ylim(0, 0.5)
        plt.ylabel('P(wing flick = 1 | state)')
        plt.xlabel('State')
        plt.xticks(xt, xt + 1)
        plt.tight_layout()
        if savefig: plt.savefig(os.path.join(fig_dir, f'{s}_skip_states={len(skip_states)}_precomp_auxem_frac_by_z.pdf'),
                                bbox_inches='tight', dpi=300, transparent=True)
        if display: plt.show()
        plt.close()
    return


# def plot_auxem_fraction_by_z_o(acc_z_o, ay_labels, title=None, savefig=False, fig_dir=None, display=True):
#     """
#     Plot aux em event fractions in each state for each auxiliary emission dimension separately
#     """
#     fig, ax = plt.subplots(1, len(acc_z_o), figsize=(7, 4), sharey=True, layout='constrained')
#     for z in acc_z_o:
#         axes = ax[z] if len(acc_z_o) > 1 else ax
#         for o in acc_z_o[z]:
#             axes.bar(o, acc_z_o[z][o] * 100, color=COLORS[z])
#         axes.set_xticks(list(acc_z_o[z].keys()), list(ay_labels.values()), rotation=0)
#         axes.set_title(f'State {z+1}', color=COLORS[z])
#         axes.axhline(0, c='k', ls=':', lw=2)
#         axes.margins(0.1)
#
#     axes = ax[0] if len(acc_z_o) > 1 else ax
#     axes.set_ylabel('Fraction of behavior (%)')
#     # plt.suptitle(title)
#     plt.ylim(0, 100)
#     plt.tight_layout()
#     if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_auxem_fraction_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
#     if display: plt.show()
#     plt.close()
#     return fig


def plot_var_explained_by_z_o_by_fly(r2_z_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot r2 scores in each state for each emission dimension separately, by fly
    """
    # print("o_labels", o_labels)
    figwidth = 3*len(r2_z_o)+0.5
    fig, ax = plt.subplots(1, len(r2_z_o), figsize=(figwidth, 7), sharey=True, layout='constrained')

    sort_by = np.argsort(r2_z_o[0][0])[::-1]
    colors = np.linspace(0, 1, len(r2_z_o[0][0]))
    n_colors = len(r2_z_o[0][0])

    for z in r2_z_o:
        axes = ax[z] if len(r2_z_o) > 1 else ax

        base_rgba = to_rgba(COLORS[z], alpha=1.0)
        faded_colors = np.tile(base_rgba, (n_colors, 1))
        faded_colors[:, -1] = np.linspace(1, 0.3, n_colors)
        transparent_cmap = ListedColormap(faded_colors)

        for o in r2_z_o[z]:
            scores = r2_z_o[z][o] * 100
            jitter = np.random.uniform(-0.1, 0.1, len(scores))
            axes.scatter(o + jitter, scores[sort_by], c=colors, cmap=transparent_cmap, s=SCATTERSIZE)
            axes.errorbar(o + 0.2, np.mean(scores), yerr=np.std(scores), color='k', alpha=0.5, fmt='o', capsize=0)

        axes.set_xticks(list(r2_z_o[z].keys()), list(o_labels.values()), rotation=0)
        axes.set_title(f'State {z+1}', color=COLORS[z])
        axes.axhline(0, c='k', ls=':', lw=2)
        axes.margins(0.1)

    axes = ax[0] if len(r2_z_o) > 1 else ax
    axes.set_ylabel('Var explained (%)')
    # plt.suptitle(title)
    # plt.ylim(-10, 20)
    # plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_z_and_o_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    return fig


def plot_pearson_by_z_o_by_fly(pearson_z_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot pearson in each state for each emission dimension separately, by fly
    """
    # print("o_labels", o_labels)
    fig_width = 3*len(pearson_z_o)+0.5
    fig, ax = plt.subplots(1, len(pearson_z_o), figsize=(fig_width, 7), sharey=True, layout='constrained')

    sort_by = np.argsort(pearson_z_o[0][0])[::-1]
    colors = np.linspace(0, 1, len(pearson_z_o[0][0]))
    n_colors = len(pearson_z_o[0][0])

    for z in pearson_z_o:
        axes = ax[z] if len(pearson_z_o) > 1 else ax

        base_rgba = to_rgba(COLORS[z], alpha=1.0)
        faded_colors = np.tile(base_rgba, (n_colors, 1))
        faded_colors[:, -1] = np.linspace(1, 0.3, n_colors)
        transparent_cmap = ListedColormap(faded_colors)

        for o in pearson_z_o[z]:
            scores = pearson_z_o[z][o]
            jitter = np.random.uniform(-0.1, 0.1, len(scores))
            axes.scatter(o + jitter, scores[sort_by], c=colors, cmap=transparent_cmap, s=SCATTERSIZE, edgecolors='none')
            axes.errorbar(o + 0.2, np.mean(scores), yerr=np.std(scores), color='k', alpha=0.5, fmt='o', capsize=0)

        axes.set_xticks(list(pearson_z_o[z].keys()), list(o_labels.values()), rotation=0)
        axes.set_title(f'State {z+1}', color=COLORS[z])
        axes.axhline(0, c='k', ls=':', lw=2)
        axes.margins(0.1)

    axes = ax[0] if len(pearson_z_o) > 1 else ax
    axes.set_ylabel(r"Pearson $r$")
    # plt.suptitle(title)
    # plt.ylim(-10, 20)
    # plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_pearson_by_z_and_o_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
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
    plt.title(f'{prefix.title()}:{batch} {suffix}')
    plt.xlabel('Time (s)')
    # plt.legend(loc='upper right')

    plt.tight_layout()
    if savefig: fig.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_comparison_probs(state_probs1, state_probs2, config, batch, effective_fps, xlim=None, xlim_orig=None, prefix_data='', suffix='', savefig=False, fig_path=None, display=True):
    xlim_ = np.r_[xlim[0]:xlim[1]]

    fig = plt.figure(figsize=(20, 4))
    for z in range(config['num_states']):
        # Probability of z at each time step
        # plt.plot(range(state_probs.shape[1]), uniform_filter1d(state_probs[batch][:, z], size=100), c=COLORS[z], linewidth=1.5, label=f'State {z}')
        plt.plot(xlim_/effective_fps, state_probs1[batch][xlim_, z], c=COLORS[z], linewidth=3, label=f'State {z+1}')
        plt.plot(xlim_/effective_fps, state_probs2[batch][xlim_, z], '.-', c=COLORS[z], linewidth=1, label=f'State {z+1}')

    # plt.plot(uniform_filter1d(state_probs[batch], size=500, axis=0))
    plt.ylim([-0.05, 1.05])
    plt.xlim(xlim_[0]/effective_fps, xlim_[-1]/effective_fps)
    plt.yticks([0, 1], fontsize='x-large')
    plt.xticks([])
    plt.legend(loc='upper right')
    plt.ylabel('P(state | data)', fontsize='x-large')
    plt.title(f'Example session {prefix_data.title()}:{batch} ({suffix})')
    # plt.xlabel('Time (s)')
    # plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    if savefig: fig.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_trajectories(model_ckp, model_config, data_config, batch, prefix='', suffix='', states_in_bgr=True, xlim=None, xlim_orig=None, savefig=False, fig_path=None, display=True):

    # num_states = model_config['num_states']
    model_label = model_ckp['prefix'].upper() #+ '_' + str(num_states)
    emission_labels_jr = data_config['emission_labels_jr']
    emission_labels_zscored = data_config['emission_labels_zscored']

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
    print("emissions_per_state", emissions_per_state.shape)
    true_emissions = model_ckp[f'{prefix}_data'][f'{prefix}_emissions'][batch]
    stateseq = model_ckp[f'{prefix}_data'][f'{prefix}_stateseq'][batch]

    for o_index in range(emissions.shape[-1]):
        plot_hmm_data_whole_session_perstate_states_on_top(
            emissions, emissions_per_state, true_emissions, stateseq, data_config, model_config, o_index=o_index, model_label=model_label, y_labels=emission_labels_zscored, xlim=xlim, xlim_orig=xlim_orig,
            savefig=savefig, fig_path=f'{fig_path}_o_index={str(o_index)}.pdf', display=display)
    return


def plot_trajectories_w_partner(model_ckp, model_config, data_config, batch, prefix='', suffix='', xlim=None, xlim_orig=None, savefig=False, fig_path=None, display=True):

    model_label = model_ckp['prefix'].upper()
    emission_labels = data_config['emission_labels']
    emission_labels_zscored = data_config['emission_labels_zscored']
    auxiliary_labels = data_config['auxiliary_labels']

    emissions = model_ckp[f'{prefix}_data'][f'{prefix}_soft_predictions'][batch]
    true_emissions = model_ckp[f'{prefix}_data'][f'{prefix}_emissions'][batch]
    stateseq = model_ckp[f'{prefix}_data'][f'{prefix}_stateseq'][batch]
    aux_data = model_ckp[f'{prefix}_data'][f'{prefix}_aux_data'][batch]

    plot_hmm_data_whole_session_with_aux_with_states(
        emissions, true_emissions, aux_data, stateseq, data_config, model_label=model_label, y_labels=emission_labels_zscored, aux_labels=auxiliary_labels, xlim=xlim, xlim_orig=xlim_orig,
        title=f'Sensory inputs and predicted trajectory ({prefix.capitalize()}:{batch}) {suffix}', savefig=savefig, fig_path=fig_path, display=display)
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


def plot_trajectories2D(model_ckp, model_config, data_config, batch, prefix='', suffix='', states_in_bgr=True, xlim=None, xlim_orig=None, savefig=False, fig_path=None, display=True):


    predicted_emissions = model_ckp[f'{prefix}_data'][f'{prefix}_soft_predictions'][batch]
    true_emissions = model_ckp[f'{prefix}_data'][f'{prefix}_emissions'][batch]

    plt.figure()

    xlim_ = np.r_[xlim[0]:xlim[1] + 1].astype(int)
    dt = 1 / data_config['effective_fps']

    fFV = true_emissions[xlim_, 0]
    fLV = true_emissions[xlim_, 1]
    fRV = true_emissions[xlim_, 2]
    x, y = 0.0, 0.0
    theta = 0.0  # initial heading angle
    trajectory_x = []
    trajectory_y = []
    for i in range(len(fFV)):
        theta += np.deg2rad(np.abs(fRV[i])) * dt
        x += (fFV[i] * np.cos(theta) - fLV[i] * np.sin(theta)) * dt
        y += (fFV[i] * np.sin(theta) + fLV[i] * np.cos(theta)) * dt
        trajectory_x.append(x)
        trajectory_y.append(y)

    fFV = predicted_emissions[xlim_, 0]
    fLV = predicted_emissions[xlim_, 1]
    fRV = predicted_emissions[xlim_, 2]
    x, y = 0.0, 0.0
    theta = 0.0  # initial heading angle
    trajectory_model_x = []
    trajectory_model_y = []
    for i in range(len(fFV)):
        theta += np.deg2rad(np.abs(fRV[i])) * dt
        x += (fFV[i] * np.cos(theta) - fLV[i] * np.sin(theta)) * dt
        y += (fFV[i] * np.sin(theta) + fLV[i] * np.cos(theta)) * dt
        trajectory_model_x.append(x)
        trajectory_model_y.append(y)

    plt.plot(trajectory_x, trajectory_y, label='Data', color='gray')
    plt.plot(trajectory_model_x, trajectory_model_y, label='GLM-HMM', color=EC)
    plt.plot(0, 0, 'o', color='k')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend(loc='upper right')
    plt.title('Reconstructed Female Trajectory')
    plt.tight_layout()

    if savefig: plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=False)
    if display: plt.show()
    plt.close()

    return


def plot_STAs(model_ckp, model_config, data_config, prefix='', savefig=False, fig_path=None, display=True):
    """state-triggered averages"""

    # predicted_emissions = np.concatenate(model_ckp[f'{prefix}_data'][f'{prefix}_soft_predictions'], axis=0)
    true_emissions = np.concatenate(model_ckp[f'{prefix}_data'][f'{prefix}_emissions'], axis=0)
    aux_data = np.concatenate(model_ckp[f'{prefix}_data'][f'{prefix}_aux_data'], axis=0)
    states = np.concatenate(model_ckp[f'{prefix}_data'][f'{prefix}_stateseq'], axis=0)

    a_labels = data_config['auxiliary_labels']
    o_labels = data_config['emission_labels']

    mfv = aux_data[:, list(a_labels.keys()).index('mFV')]
    ffv = true_emissions[:, list(o_labels.keys()).index('fFV')]

    # parameters
    effective_fps = data_config['effective_fps']
    window_s = 0.25  # window size (seconds) before and after switch
    min_dwell_s = 0.25  # minimum dwell time (seconds)
    window_frames = int(window_s * effective_fps)
    min_dwell_frames = int(min_dwell_s * effective_fps)

    # --- STEP 1: Find switch points ---
    switch_points = np.where(states[1:] != states[:-1])[0] + 1
    print(switch_points)

    # --- STEP 2: Filter switches by minimum dwell time ---
    clean_switches = []
    for sp in switch_points:
        # check dwell before
        pre_state = states[sp - 1]
        pre_dwell = 0
        i = sp - 1
        while i >= 0 and states[i] == pre_state:
            pre_dwell += 1
            i -= 1

        # check dwell after
        post_state = states[sp]
        post_dwell = 0
        i = sp
        while i < len(states) and states[i] == post_state:
            post_dwell += 1
            i += 1

        if pre_dwell >= min_dwell_frames and post_dwell >= min_dwell_frames:
            clean_switches.append(sp)

    clean_switches = np.array(clean_switches)
    print("clean_switches", len(clean_switches))

    from_states = states[clean_switches - 1]
    to_states = states[clean_switches]

    # target_state = 3  # A
    # keep_idx = np.where(to_states == target_state)[0]
    # selected_switches = switch_points[keep_idx]

    plt.figure(figsize=(15, 8))

    for z in [-1]:

        keep_idx = np.where((to_states == 4))[0]
        selected_switches = clean_switches[keep_idx]
        print(f"{z} selected_switches", len(selected_switches))
        if not len(selected_switches): continue

        # --- STEP 3: Extract windows ---
        ffv_windows = []
        mfv_windows = []
        for sp in selected_switches:
            start = max(0, sp - window_frames)
            end = min(len(ffv), sp + window_frames)

            # pad if near edges
            pad_left = window_frames - (sp - start)
            pad_right = window_frames - (end - sp)

            ffv_win = ffv[start:end]
            mfv_win = mfv[start:end]

            if pad_left > 0:
                ffv_win = np.pad(ffv_win, (pad_left, 0), mode='edge')
                mfv_win = np.pad(mfv_win, (pad_left, 0), mode='edge')
            if pad_right > 0:
                ffv_win = np.pad(ffv_win, (0, pad_right), mode='edge')
                mfv_win = np.pad(mfv_win, (0, pad_right), mode='edge')

            ffv_windows.append(ffv_win)
            mfv_windows.append(mfv_win)

        # --- STEP 4: Compute average ---
        ffv_mean = np.mean(ffv_windows, axis=0)
        ffv_sem = np.std(ffv_windows, axis=0) / np.sqrt(len(ffv_windows))
        mfv_mean = np.mean(mfv_windows, axis=0)
        mfv_sem = np.std(mfv_windows, axis=0) / np.sqrt(len(mfv_windows))

        time_axis = np.linspace(-window_s, window_s, 2 * window_frames)

        # --- STEP 5: Plot ---
        plt.plot(time_axis, ffv_mean, label=f'fFV (state {z+1}) ({len(selected_switches)})', color=COLORS[z])
        plt.fill_between(time_axis, ffv_mean - ffv_sem, ffv_mean + ffv_sem, alpha=0.3, color=COLORS[z])

        plt.plot(time_axis, mfv_mean, ':', label=f'mFV (state {z+1})', color=COLORS[z])
        plt.fill_between(time_axis, mfv_mean - mfv_sem, mfv_mean + mfv_sem, alpha=0.2, color=COLORS[z])

    plt.axvline(0, color='gray', linestyle='--', label='State switch')
    plt.xlabel('Time around switch (s)')
    plt.ylabel('Velocity')
    plt.legend(loc='upper left')
    plt.title(f'Behavior around state switches (min dwell {min_dwell_s}s)')
    plt.tight_layout()
    plt.show()

    return


def plot_ETSPs(model_ckp, model_config, data_config, savefig=False, fig_path=None, display=True):
    """event-triggered state probs"""

    aux_data = np.concatenate([*model_ckp[f'train_data'][f'train_aux_data'], *model_ckp[f'test_data'][f'test_aux_data']], axis=0)
    states = np.concatenate([*model_ckp[f'train_data'][f'train_stateseq'], *model_ckp[f'test_data'][f'test_stateseq']], axis=0)

    a_labels = data_config['auxiliary_labels']

    song = aux_data[:, list(a_labels.keys()).index('pfast_i')]

    # parameters
    effective_fps = data_config['effective_fps']
    window_s = 0.25  # window size (seconds) before and after switch
    min_duration_s = 0.2  # minimum dwell time (seconds)
    window_frames = int(window_s * effective_fps)
    min_duration_frames = int(min_duration_s * effective_fps)

    # --- STEP 1: Find song onsets ---
    song = np.where(song > 0, 1, 0)
    # print(song.shape, np.max(song), np.min(song))
    # plt.figure(figsize=(20, 5))
    # plt.plot(song)
    # plt.axhline()
    # plt.show()
    # song_onsets = np.where((song[1:] == 1) & (song[:-1] == 0))[0] + 1

    song_onsets = np.where((song[1:] == 1) & (song[:-1] == 0))[0] + 1
    song_offsets = np.where((song[1:] == 0) & (song[:-1] == 1))[0] + 1
    if song[0] == 1:
        song_onsets = np.insert(song_onsets, 0, 0)
    if song[-1] == 1:
        song_offsets = np.append(song_offsets, len(song) - 1)
    durations = song_offsets - song_onsets  # in frames
    keep_idx = np.where(durations >= min_duration_frames)[0]
    filtered_onsets = song_onsets[keep_idx]

    # --- STEP 2: Extract state windows ---
    state_windows = []
    for onset in filtered_onsets:
        start = max(0, onset - window_frames)
        end = min(len(states), onset + window_frames)
        pad_left = window_frames - (onset - start)
        pad_right = window_frames - (end - onset)

        win = states[start:end]
        if pad_left > 0:
            win = np.pad(win, (pad_left, 0), mode='edge')
        if pad_right > 0:
            win = np.pad(win, (0, pad_right), mode='edge')

        state_windows.append(win)
    state_windows = np.stack(state_windows)  # shape (n_onsets, 2 * window_frames)
    print("state_windows", state_windows)

    # --- STEP 3: Compute state probabilities ---
    N_states = model_config['num_states']
    time_axis = np.linspace(-window_s, window_s, 2 * window_frames)

    plt.figure(figsize=(15, 8))
    for z in range(N_states):
        # prob_matrix_all = np.array([state_windows == s for s in range(N_states)])
        prob_mean = np.mean(state_windows == z, axis=0)
        prob_sem = np.std(state_windows == z, axis=0) / np.sqrt(len(state_windows))

        plt.plot(time_axis, prob_mean, label=f'State {z+1}', color=COLORS[z])
        plt.fill_between(time_axis, prob_mean - prob_sem, prob_mean + prob_sem, alpha=0.2, color=COLORS[z])

    plt.axvline(0, color='gray', linestyle='--', label='Song onset')
    plt.xlabel('Time around song onset (s)')
    plt.ylabel('Probability')
    plt.legend(loc='upper left')
    plt.title(f'Female state probabilities around {len(filtered_onsets)} male song onsets')
    plt.tight_layout()
    plt.show()

    return


def rescale_aux_data(aux_data, aux_mn_std):
    """
    :param aux_data:
    :param aux_mn_std:
    :return:
    """
    aux_data_rescaled = []
    for btch in range(len(aux_data)):
        mn_std_btch = aux_mn_std[btch]
        eez = aux_data[btch] * mn_std_btch[:, 1, None].T + mn_std_btch[:, 0, None].T
        aux_data_rescaled.append(eez)
    return aux_data_rescaled


def rescale_o_data(o_data, output_mn_std):
    """
    :param o_data:
    :param output_mn_std:
    :return:
    """
    o_data_rescaled = []
    for btch in range(len(o_data)):
        mn_std_btch = output_mn_std[btch]
        eez = o_data[btch] * mn_std_btch[:, 1, None].T + mn_std_btch[:, 0, None].T
        o_data_rescaled.append(eez)
    return o_data_rescaled


def plot_ETAs(model_ckp, model_config, data_config, savefig=False, fig_dir=None, display=True):
    """event-triggered averages"""

    model_prefix = model_ckp['prefix']
    all_aux_data = [*model_ckp[f'train_data'][f'train_aux_data'], *model_ckp[f'test_data'][f'test_aux_data']]
    all_aux_mn_std = [*model_ckp['train_data']['train_aux_mn_std'], *model_ckp['test_data']['test_aux_mn_std']]
    # print(len(all_aux_mn_std), len(all_aux_data), all_aux_data[0].shape, all_aux_data[0][:, -1])
    all_aux_data_rescaled = rescale_aux_data(all_aux_data, all_aux_mn_std)
    # print(len(all_aux_data_rescaled), all_aux_data_rescaled[0].shape, all_aux_data_rescaled[0][:, -1])
    all_aux_data_rescaled = np.concatenate(all_aux_data_rescaled, axis=0)
    true_emissions = np.concatenate([*model_ckp[f'train_data'][f'train_emissions'], *model_ckp[f'test_data'][f'test_emissions']], axis=0)
    pred_emissions = np.concatenate([*model_ckp[f'train_data'][f'train_soft_predictions'], *model_ckp[f'test_data'][f'test_soft_predictions']], axis=0)

    a_labels = data_config['auxiliary_labels_full']
    a_labels_list = data_config['auxiliary_labels_list']
    o_labels = data_config['emission_labels_zscored']

    o_labels['fLV'] = '|lateral velocity|\n(zscored)'    # to use in ETAs where left and right data will otherwise cancel each other out.
    o_labels['fAV'] = '|angular velocity|\n(zscored)'

    # parameters
    effective_fps = data_config['effective_fps']
    window_s = 0.5  # window size (seconds) before and after switch
    window_frames = int(window_s * effective_fps)

    time_axis = np.linspace(-window_s, window_s, 2 * window_frames)
    emission_dim = true_emissions.shape[-1]

    for event_name in ['pulse_i', 'sine_i', 'tap2']:

        # --- STEP 1: Find event onsets ---
        event_ts = all_aux_data_rescaled[:, a_labels_list.index(event_name)]
        # fmAng_sin_ts = all_aux_data_rescaled[:, a_labels_list.index('fmAng_sin')]
        label = a_labels[event_name]
        min_duration_s = 0.1 if event_name in ['pulse_i', 'sine_i'] else 0.01  # minimum event duration (seconds)
        print(event_name, f"min_duration_s={min_duration_s}")
        filtered_onsets = get_event_onsets(event_ts, min_duration=int(min_duration_s * effective_fps), lr_mask=None)

        fig, axes = plt.subplots(1, emission_dim, figsize=(4.5*emission_dim+1+2, 5), sharex=True)
        for o, ol in enumerate(o_labels):

            true_feat_series = true_emissions[:, o]
            pred_feat_series = pred_emissions[:, o]

            if 'LV' in ol or 'AV' in ol:
                true_feat_series = np.abs(true_feat_series)
                pred_feat_series = np.abs(pred_feat_series)

            ax = axes[o]

            # --- STEP 2: Extract feat windows ---
            true_feat_series_windows, pred_feat_series_windows = get_feat_windows(true_feat_series, pred_feat_series, filtered_onsets, window_frames)

            # --- STEP 3: Compute averages ---
            prob_mean = np.mean(true_feat_series_windows, axis=0)
            prob_sem = np.std(true_feat_series_windows, axis=0) / np.sqrt(len(true_feat_series_windows))

            prob_mean_p = np.mean(pred_feat_series_windows, axis=0)
            prob_sem_p = np.std(pred_feat_series_windows, axis=0) / np.sqrt(len(pred_feat_series_windows))

            print("n=", len(true_feat_series_windows))

            ax.plot(time_axis, prob_mean, label=f'Data', color='gray', lw=2)
            ax.fill_between(time_axis, prob_mean - prob_sem, prob_mean + prob_sem, alpha=0.2, color='gray')
            ax.plot(time_axis, prob_mean_p, label=model_prefix.upper(), color=EC, lw=2)
            ax.fill_between(time_axis, prob_mean_p - prob_sem_p, prob_mean_p + prob_sem_p, alpha=0.2, color=EC)
            ax.axvline(0, color=input_label_colors[event_name], linestyle='--', lw=2, label=f'{label}\nonset')
            ax.set_ylabel(o_labels[ol])
            # ax.axvspan(0, min_duration_s, color=input_label_colors[event_name], alpha=0.3, label=f'{label}')
            if ax.get_subplotspec().is_last_col():
                ax.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.5, 1), borderaxespad=0.)
            ax.margins(0.1)

        fig.supxlabel('Time relative to event onset (s)', fontsize='large')
        # fig.suptitle(f'Female behavior around {label} onsets (n={len(true_feat_series_windows)})', fontsize='large')
        plt.tight_layout()
        if savefig: fig.savefig(os.path.join(fig_dir, f'ETAs_{event_name}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
        if display: plt.show()
        plt.close()
    return


def plot_ETAs_all(model_ckp, data_config, savefig=False, fig_dir=None, display=True):

    def construct_traces(fFV, fLV, fAV, dt):
        """
        Batched integration of N trajectories from body-centered velocities.
        Inputs:
            fwd (N, T): forward velocity
            lat (N, T): lateral velocity
            ang (N, T): angular velocity (radians/sec)
            dt (float): time step size
        Returns:
            x, y, theta: arrays of shape (N, T+1)
        """
        N, T = fFV.shape
        x = np.zeros((N, T + 1))
        y = np.zeros((N, T + 1))
        theta = np.zeros((N, T + 1))

        x[:, 0] = 0.
        y[:, 0] = 0.
        theta[:, 0] = 0.

        for t in range(T):
            theta[:, t + 1] = theta[:, t] + fAV[:, t] * dt
            dx = (fFV[:, t] * np.cos(theta[:, t]) - fLV[:, t] * np.sin(theta[:, t])) * dt
            dy = (fFV[:, t] * np.sin(theta[:, t]) + fLV[:, t] * np.cos(theta[:, t])) * dt
            x[:, t + 1] = x[:, t] + dx
            y[:, t + 1] = y[:, t] + dy
        traces = np.stack([x, y, theta], axis=-1)
        return traces

    def plot(masked_trajs_windows, dir=None):
        color = LR[dir]
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        mean_traj = np.mean(masked_trajs_windows, axis=0)
        print("mean_traj", mean_traj.shape, "masked_trajs_windows", mean_traj.shape)
        r = np.random.choice(len(masked_trajs_windows), size=min(200, len(masked_trajs_windows)))
        ax.plot(masked_trajs_windows[r, :, 0].T, masked_trajs_windows[r, :, 1].T, '-', alpha=0.1, color=color)
        ax.plot(mean_traj[:, 0].T, mean_traj[:, 1].T, '-', alpha=1, lw=2, color=color)
        ax.set_aspect('equal')
        ax.axhline(0, lw=2, ls=':', c='k')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.axis('off')
        plt.tight_layout()
        if savefig: fig.savefig(os.path.join(fig_dir, f'ETAs_{event_name}_{dir}.pdf'), bbox_inches='tight', dpi=300,
                                transparent=True)
        if display: plt.show()
        plt.close()
        return

    all_aux_data = [*model_ckp[f'train_data'][f'train_aux_data'], *model_ckp[f'test_data'][f'test_aux_data']]
    all_aux_mn_std = [*model_ckp['train_data']['train_aux_mn_std'], *model_ckp['test_data']['test_aux_mn_std']]
    all_aux_data_rescaled = rescale_aux_data(all_aux_data, all_aux_mn_std)
    all_aux_data_rescaled = np.concatenate(all_aux_data_rescaled, axis=0)

    all_outputs = [*model_ckp[f'train_data'][f'train_emissions'], *model_ckp[f'test_data'][f'test_emissions']]
    all_output_mn_std = [*model_ckp['train_data']['train_output_mn_std'], *model_ckp['test_data']['test_output_mn_std']]
    all_outputs_rescaled = rescale_o_data(all_outputs, all_output_mn_std)
    all_outputs_rescaled = np.concatenate(all_outputs_rescaled, axis=0)

    a_labels_list = data_config['auxiliary_labels_list']

    # parameters
    effective_fps = data_config['effective_fps']
    window_s = 1  # window size (seconds) before and after switch
    window_frames = int(window_s * effective_fps)

    for event_name in ['pulse_i', 'sine_i']:

        # --- STEP 1: Find event onsets ---
        event_ts = all_aux_data_rescaled[:, a_labels_list.index(event_name)]
        fmAng_sin_ts = all_aux_data_rescaled[:, a_labels_list.index('fmAng_sin')]

        min_duration_s = 0.1 if event_name in ['pulse_i', 'sine_i'] else 0.01  # minimum event duration (seconds)
        print(event_name, f"min_duration_s={min_duration_s}")
        filtered_onsets = get_event_onsets(event_ts, min_duration=int(min_duration_s * effective_fps), lr_mask=None)

        fmAng_sin_windows, _ = get_feat_windows(fmAng_sin_ts, fmAng_sin_ts, filtered_onsets, window_frames)
        fFV_windows, _ = get_feat_windows(all_outputs_rescaled[:, 0], all_outputs_rescaled[:, 0], filtered_onsets, window_frames)
        fLV_windows, _ = get_feat_windows(all_outputs_rescaled[:, 1], all_outputs_rescaled[:, 1], filtered_onsets, window_frames)
        fAV_windows, _ = get_feat_windows(all_outputs_rescaled[:, 2], all_outputs_rescaled[:, 2], filtered_onsets, window_frames)

        fmAng_sin_windows = fmAng_sin_windows[:, -window_frames:]  # remove the times before onset
        fFV_windows = fFV_windows[:, -window_frames:]
        fLV_windows = fLV_windows[:, -window_frames:]
        fAV_windows = fAV_windows[:, -window_frames:]

        traj_windows = construct_traces(fFV_windows, fLV_windows, np.deg2rad(fAV_windows), effective_fps)

        fmAng_windows = np.rad2deg(np.arcsin(fmAng_sin_windows))
        print(fmAng_sin_windows.shape, fmAng_windows.shape, fmAng_windows[0])
        print(fFV_windows.shape, fFV_windows[0])
        print("trajs_windows", traj_windows.shape)

        # male on left
        left_mask = ((fmAng_windows[:, 0] > 60) & (fmAng_windows[:, 0] < 90)).squeeze()
        plot(traj_windows[left_mask], 'L')
        # male on right
        right_mask = ((fmAng_windows[:, 0] < -60) & (fmAng_windows[:, 0] > -90)).squeeze()
        plot(traj_windows[right_mask], 'R')
    return
