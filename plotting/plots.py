import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.ticker import FixedLocator
import seaborn as sns
import os
import networkx as nx

import numpy as np
# import jax.numpy as jnp
# import jax.random as jr
from dynamax.utils.plotting import CMAP, COLORS
from scipy.ndimage import uniform_filter1d
from glm_utils.preprocessing import BasisProjection


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
mpl.rcParams['axes.linewidth'] = 0.7
mpl.rcParams['axes.ymargin'] = 0
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14

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

# -- Fonts --
mpl.rcParams['font.size'] = 16  # Panel label
# mpl.rcParams['font.family'] = 'Arial'
# mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'

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
        ax.set_ylabel(y_labels[_], c='magenta')
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
        ax.set_ylabel(y_labels[_], fontsize='xx-small', c='magenta')
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


def plot_filters_statewise(weights, data_config, y_labels, prefix, savefig=False, fig_dir=None, display=True):
    # print(weights.shape)

    num_states = weights.shape[0]
    emission_dim = weights.shape[1]
    filter_len = weights.shape[-1]
    input_labels = data_config['input_labels']
    n_inputs = len(input_labels)
    basis = data_config['basis']

    if basis is not None:
        weights = BasisProjection(basis).inverse_transform(weights.reshape(-1, filter_len)).reshape(num_states, emission_dim, -1)
    weights = weights.reshape(num_states, emission_dim, n_inputs, -1)
    weights = weights/np.linalg.norm(weights)

    fig, axs = plt.subplots(emission_dim, num_states, figsize=(2.4 * num_states, 2.4 * len(y_labels)), sharey=True)
    d = 0
    for _ in y_labels:
        w = weights[:, d]
        for z in range(num_states):

            if emission_dim > 1 and num_states > 1:
                ax = axs[d, z]
            elif emission_dim > 1 and num_states == 1:
                ax = axs[d]
            elif emission_dim == 1 and num_states > 1:
                ax = axs[z]
            else:
                ax = axs

            if ax.get_subplotspec().is_first_row():
                ax.set_title(f'State {z + 1}', fontsize='large', color=COLORS[z])
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(f'{y_labels[_]}\nfilter amplitude (a.u.)', fontsize='x-small')

            stim = 0
            for __ in input_labels:
                if __ not in ['mFV', 'side', 'pfast_i', 'tap2']:
                    stim += 1
                    continue
                ax.plot(np.arange(-w[z, stim].shape[-1], 0)/data_config['orig_fps'], w[z, stim], linewidth=3, label=input_labels[__])
                ax.axhline(0, ls=':', c='k', lw=0.5)
                ax.set_xticks([-w[0, stim].shape[-1]//data_config['orig_fps'], 0])
                ax.margins(y=0.05)
                if ax.get_subplotspec().is_last_row():
                    ax.set_xlabel('Time (s)')
                stim += 1

        d += 1

    if emission_dim > 1 and num_states > 1:
        fig.align_ylabels(axs[:, 0])
    elif emission_dim > 1 and num_states == 1:
        fig.align_ylabels(axs[:])

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{prefix}_filters_statewise.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_filters(weights, data_config, y_labels, filesuffix, savefig=False, fig_dir=None, display=True):
    # print(weights.shape)

    num_states = weights.shape[0]
    emission_dim = weights.shape[1]
    filter_len = weights.shape[-1]
    input_labels = data_config['input_labels']
    n_inputs = len(input_labels)
    basis = data_config['basis']

    if basis is not None:
        weights = BasisProjection(basis).inverse_transform(weights.reshape(-1, filter_len)).reshape(num_states, emission_dim, -1)
    weights = weights.reshape(num_states, emission_dim, n_inputs, -1)
    weights = weights/np.linalg.norm(weights)

    fig, axs = plt.subplots(emission_dim, n_inputs, figsize=(21, 2.25 * len(y_labels)), sharey=True)
    d = 0
    for _ in y_labels:
        w = weights[:, d]
        stim = 0
        for __ in input_labels:
            ax = axs[d, stim] if emission_dim > 1 else axs[stim]

            for z in range(num_states):
                ax.plot(np.arange(-w[z, stim].shape[-1], 0)/data_config['orig_fps'], w[z, stim], color=COLORS[z], linewidth=3, label=f'State {z+1}')

            if d == 0:
                ax.set_title(f'{input_labels[__]}', fontsize='large')
            if stim == 0:
                ax.set_ylabel(f'{y_labels[_]}\nfilter amplitude (a.u.)', fontsize='x-small', color='magenta')

            ax.axhline(0, ls=':', c='k', lw=0.5)
            ax.set_xticks([-w[0, stim].shape[-1]//data_config['orig_fps'], 0])
            ax.margins(y=0.05)
            # ax.set_ylim(-0.1, 0.1)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel('Time (s)')
            stim += 1

        d += 1

    # ax = axs[0, 0] if emission_dim > 1 else axs[0]
    # ax.legend(loc='upper left')
    # fig.supxlabel("Time relative to prediction (s)")
    if emission_dim > 1: fig.align_ylabels(axs[:, 0])
    plt.margins(0.02)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'filters_{filesuffix}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_filter_amplitudes(weights, data_config, y_labels, prefix, plot_top_k=None, savefig=False, fig_dir=None, display=True):
    # print(weights.shape)

    num_states = weights.shape[0]
    emission_dim = weights.shape[1]
    filter_len = weights.shape[-1]
    input_labels = data_config['input_labels']
    n_inputs = len(input_labels)
    basis = data_config['basis']

    if basis is not None:
        weights = BasisProjection(basis).inverse_transform(weights.reshape(-1, filter_len)).reshape(num_states, emission_dim, -1)
    weights = weights.reshape(num_states, emission_dim, n_inputs, -1)
    weights = weights/np.linalg.norm(weights)

    if plot_top_k:
        fig_width = 10
    else:
        fig_width = 15
    fig, axs = plt.subplots(num_states, emission_dim, figsize=(fig_width, 10), sharey=True)
    xticklabels = np.array([_ for _ in list(input_labels.values())])
    # print("weights.shape", weights.shape)

    d = 0
    for _ in y_labels:
        for z in range(num_states):
            w_l2 = np.linalg.norm(weights[z][d], axis=-1)
            w_l2_scaled = (w_l2 - np.min(w_l2))/(np.max(w_l2) - np.min(w_l2) + 1e-5)
            # print(w_l2_scaled, w_l2.shape, w_l2_scaled.shape)
            sorted_idxs = np.argsort(w_l2_scaled)
            if plot_top_k:
                sorted_idxs = sorted_idxs[np.r_[-5:0]]  # highest weighted inputs only
            if (num_states > 1) and (emission_dim > 1):
                ax = axs[z, d]
            elif (num_states == 1) and (emission_dim > 1):
                ax = axs[d]
            elif (num_states > 1) and (emission_dim == 1):
                ax = axs[z]
            elif (num_states == 1) and (emission_dim == 1):
                ax = axs
            ax.plot(w_l2_scaled[sorted_idxs], 'k.', markersize=12)
            ax.set_xticks(range(len(sorted_idxs)), xticklabels[sorted_idxs], rotation=90, fontsize='small')
            ax.set_yticks([0, 0.5, 1], [0, '', 1])
            ax.set_ylim([-0.1, 1.1])
            ax.margins(x=0.1)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.tick_params(axis='both', direction='in', top=True, right=True)
            if d == 0: ax.set_ylabel(f'State {z + 1}', color=COLORS[z], fontsize='x-large')
            if z == 0: ax.set_title(y_labels[_], fontsize='large', color='magenta')
        d += 1

    fig.suptitle('Amplitude of filters')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{prefix}_filter_amplitudes_plot_top_k={plot_top_k}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_mean_outputs_by_z(model_config, outputs_z, data_config):
    emission_labels = data_config['emission_labels']
    num_states = model_config['num_states']

    fig, ax = plt.subplots(1, num_states, sharex=True)
    for z in range(num_states):
        # outputs_mean = np.mean(outputs_z[z], axis=0)
        o = 0
        for __ in emission_labels:
            rand_idxs = np.random.randint(len(outputs_z[z][:, o]), size=1000)
            ax[z].scatter(
                np.random.uniform(o-0.1, o+0.1, len(rand_idxs)),
                outputs_z[z][rand_idxs, o],
                c='magenta',
                s=2,
            )
            o += 1
        ax[z].set_title(f'State {z}', color=COLORS[z])
        ax[z].set_xticks(range(len(emission_labels)), list(emission_labels.values()))
        ax[z].axhline(0, ls=":", lw=2, c='k')
    ax[0].set_ylabel('female locomotion (a.u.)')
    plt.suptitle('Female output velocity samples (n=1000)')
    plt.tight_layout()
    return fig


def plot_state_mean_outputs_by_o(model_config, outputs_z, data_config):
    emission_labels = data_config['emission_labels']
    num_states = model_config['num_states']

    fig, ax = plt.subplots(1, len(emission_labels), sharex=True)
    o = 0
    for __ in emission_labels:
        for z in range(num_states):
            rand_idxs = np.random.randint(len(outputs_z[z][:, o]), size=1000)
            ax[o].scatter(
                np.random.uniform(z-0.1, z+0.1, len(rand_idxs)),
                outputs_z[z][rand_idxs, o],
                c=COLORS[z],
                s=2,
            )
        ax[o].set_title(emission_labels[__], color='magenta')
        ax[o].set_xticks(range(num_states))
        ax[o].set_xlabel('State')
        ax[o].margins(0.1)
        ax[o].axhline(0, ls=":", lw=2, c='k')
        o += 1
    plt.suptitle('Female output samples by state (n=1000)')
    plt.tight_layout()
    return fig


def plot_state_mean_outputs_by_o_dists(emissions_z, o_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(1, len(o_labels), figsize=(16, 5))

    for o, ol in enumerate(o_labels):
        x99 = 0
        x0 = 0
        ax = axes[o] if len(o_labels) > 1 else axes
        for z in list(emissions_z.keys()):
            data = np.random.choice(np.round(emissions_z[z][:, o], decimals=3), min(10000, len(emissions_z[z])), replace=False)
            x0 = min(x0, np.percentile(data, 1))
            x99 = max(x99, np.percentile(data, 99))
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
        # ax.set_xscale('symlog')  # symmetric log, can handle negative emission values with log_scale in sns.histplot can't.
        # ax.set_yscale('log')
        ax.set_xlabel(o_labels[ol], color='magenta')
        # ax.margins(y=0.1)
        ax.set_xlim([x0, x99])
        # ax.axhline(0, ls=":", lw=2, c='k')

    ax.legend(loc='upper right')
    fig.suptitle(f'Female behavioral outputs by state')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_mean_outputs_by_o_dists.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_aux_dists(aux_z, a_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, ax = plt.subplots(1, len(a_labels), figsize=(17, 4))

    for a, al in enumerate(a_labels):
        x99 = 0
        x0 = 0
        for z in list(aux_z.keys()):
            data = np.random.choice(np.round(aux_z[z][:, a], decimals=3), min(10000, len(aux_z[z])), replace=False)
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
        ax[a].set_xlabel(a_labels[al], color='k')
        # ax[a].set_yscale('log')
        ax[a].margins(y=0.1,x=0.1)
        ax[a].set_xlim([x0-0.1, x99+0.1])
        # ymin, ymax = ax[a].get_ylim()
        # ax[a].set_ylim(ymin-0.01*ymax, ymax)
        # ax[a].axhline(0, ls=":", lw=2, c='k')
    # ax[-1].legend(loc='upper right')
    fig.suptitle(f'Sensory inputs by state')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_mean_aux_dists.pdf'),
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

        ax = axes[0, z]
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
        ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

        ax = axes[1, z]
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
        ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

        ax = axes[2, z]
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
        ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_zscored_mean_aux.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_zscored_mean_aux_odists(aux_z, emissions_z, a_labels, o_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(3, len(aux_z.keys())*2, figsize=(25, 12))

    for z in list(aux_z.keys()):
        aux_means = np.mean(aux_z[z], axis=0)
        o_means = np.mean(emissions_z[z], axis=0)
        print(z, aux_means, aux_means.shape)
        print(z, o_means, o_means.shape, np.median(emissions_z[z], axis=0))
        print(emissions_z[z].shape)
        ax_col1, ax_col2 = 2*z, 2*z+1

        ax = axes[0, ax_col1]
        for a, al in enumerate(a_labels):
            print(a, al, aux_z[z][:, a].shape)
            sns.violinplot(x=a, y=aux_z[z][:, a], ax=ax, color=COLORS[z], fill=False,
                           inner='quartile',
                           cut=0,  # do not extend beyond min/max of data
                           density_norm='area',  # makes violins comparable
                           common_norm=True,
                           )
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        ax.set_xticklabels(np.array(list(a_labels.values())), rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        ax.set_title(f'State {z+1}', color=COLORS[z])
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

        ax = axes[0, ax_col2]
        for o, ol in enumerate(o_labels):
            print(z, ol, np.mean(emissions_z[z][:, o]), np.median(emissions_z[z][:, o]))
            sns.violinplot(x=o, y=emissions_z[z][:, o], ax=ax, color=COLORS[z], fill=False,
                           inner=None,  # shows IQR only (no scatter/sticks)
                           cut=0,  # do not extend beyond min/max of data
                           density_norm='area',  # makes violins comparable
                           # linewidth=0  # remove outline
                           common_norm=True,
                           )
            # sns.kdeplot(y=emissions_z[z][:, o], ax=ax, color=COLORS[z])
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        ax.set_xticklabels(np.array(list(o_labels.keys())), rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

        ax = axes[1, ax_col1]
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
        ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

        ax = axes[1, ax_col2]
        values = o_means
        print("values o_means", o_means)
        ax.bar(range(len(values)), values, color=COLORS[z])
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(np.array(list(o_labels.keys())), rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

        ax = axes[2, ax_col1]
        values = aux_means
        sorted_by = np.argsort(np.abs(values))[::-1]
        ax.bar(range(len(values)), values[sorted_by], color=COLORS[z])
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(np.array(list(a_labels.values()))[sorted_by], rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        ax.set_title(f'State {z+1}', color=COLORS[z])
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)  # keeps all tick lines
        ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])

        ax = axes[2, ax_col2]
        values = o_means
        print("values o_means", o_means)
        ax.bar(range(len(values)), values, color=COLORS[z])
        ax.axhline(0, c='k', linewidth=0.8, ls=':')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(np.array(list(o_labels.keys())), rotation=90)
        ax.set_ylabel("z-scored value")
        ax.margins(0.1)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else '' for i, tick in enumerate(yticks)])  # keeps all tick lines but label only every other y-tick

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_zscored_mean_aux_odists.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_dwell_times(dwell_times_z, num_states, effective_fps, title='', savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    durations = []
    for z in range(num_states):
        d = dwell_times_z[z] / effective_fps    # in seconds
        durations.append(d.tolist())

    import pandas as pd
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
        # cut=0,
        clip=(0, None),
        # common_norm=True,
        linewidth=3,
    )
    x = list(range(0, 4))
    plt.xlim([x[0], x[-1]])
    plt.xticks(x)
    plt.xlabel("Time (s)")
    plt.title("State residency")

    legend = ax.get_legend()
    legend.set_title(None)
    legend.set_loc('upper right')

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

    fig = plt.figure(figsize=(13, 5))

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

    plt.grid(alpha=.3)
    plt.ylim(0, 1.05)
    if xlabel: plt.xlabel(xlabel, fontsize='large')
    plt.ylabel('p(state)', fontsize='large')
    plt.legend(loc='upper left')
    if title: plt.title(f'{title} sessions')
    if xticks: plt.xticks([0, GRID-1], xticks, fontsize='medium')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_prob_states_over_time.pdf'), bbox_inches='tight', dpi=300,
                            transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_transition_matrix(transition_matrix, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    m = transition_matrix.shape[0]
    sns.heatmap(transition_matrix, annot=True, cmap='bone', cbar=True, square=True, fmt=".3f",
                vmin=0, vmax=1, ax=ax,
                xticklabels=[f'State {i+1}' for i in range(m)],
                yticklabels=[f'State {i+1}' for i in range(m)], annot_kws={'size': 'medium'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(length=0)
    plt.title('Transition Matrix')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, 'transition_matrix.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_ethogram(transition_matrix, savefig=False, fig_dir=None, display=True):
    fig = plt.figure()

    G = nx.DiGraph()
    num_states = transition_matrix.shape[0]

    # Add edges with weights
    for i in range(num_states):
        for j in range(num_states):
            if transition_matrix[i, j] > 0.005:  # Only add edges with nonzero probability
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
            node_size=600,
            font_size=15, font_weight='bold',
            edge_color='black', width=edge_widths,
            arrows=True,
            connectionstyle='arc3,rad=0.4'
            )

    # Draw edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for (u, v, d) in edges}
    # print(edge_labels, len(edge_labels))
    nx.draw_networkx_edge_labels(G, pos, font_size=15, edge_labels=edge_labels, label_pos=0.5, connectionstyle='arc3,rad=0.4')

    # plt.title("Transition Probability Graph")
    plt.tight_layout()
    plt.margins(0.1)
    if savefig: fig.savefig(os.path.join(fig_dir, 'ethogram.pdf'), bbox_inches='tight', dpi=300, transparent=True)
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
    fig = plt.figure()
    # m = steady_state_p.shape[0]
    # sns.heatmap(steady_state_p[None, :], annot=True, cmap='bone', cbar=False, square=True,
    #             fmt=".2f",
    #             xticklabels=[f'State {_}' for _ in range(m)])
    # plt.yticks([])
    # plt.title('Steady State Probabilities')

    print("steady_state_p", steady_state_p)

    for z in range(len(steady_state_p)):
        plt.bar(z+1, steady_state_p[z] * 100, color=COLORS[z])
    plt.ylabel('Time spent\nin State(%)')
    plt.margins(0.1)
    # plt.ylim(0, 50)
    plt.xticks(range(1, 1+len(steady_state_p)))
    plt.xlabel('State')
    plt.title('Expected long-run state occupancy')
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

    fig = plt.figure()
    for z in ps_z:

        ps_z[z] = np.array(ps_z[z]) * 100
        base_rgba = to_rgba(COLORS[z], alpha=1.0)
        faded_colors = np.tile(base_rgba, (n_colors, 1))
        faded_colors[:, -1] = np.linspace(1, 0.3, n_colors)
        transparent_cmap = ListedColormap(faded_colors)

        jitter = np.random.uniform(-0.1, 0.1, len(ps_z[z]))
        plt.scatter(z+1+jitter, ps_z[z][sort_by], c=colors, cmap=transparent_cmap, s=20)
        plt.errorbar(z+1.2, np.mean(ps_z[z]), yerr=np.std(ps_z[z]), color='k', alpha=0.5, fmt='o', capsize=0)

    plt.ylabel('Time spent\nin State(%)')
    plt.margins(0.1)
    # plt.ylim(0, 50)
    plt.xticks(range(1, 1 + len(ps_z)))
    plt.xlabel('State')
    plt.title(f'Empirical state occupancy')
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
    fig = plt.figure(figsize=(3, 4))

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
    plt.title(title)
    # plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_z.pdf'), bbox_inches='tight', dpi=300, transparent=True)
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
        plt.scatter(z + jitter, r2_z[z][sort_by] * 100, c=colors, cmap=transparent_cmap, s=20)
        plt.errorbar(z + 1.2, np.mean(r2_z[z] * 100), yerr=np.std(r2_z[z] * 100), color='k', alpha=0.5, fmt='o', capsize=0)

    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks(np.array(list(r2_z.keys())).astype(int) + 1)
    plt.xlabel('State')
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    # plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_z_by_fly.pdf'), bbox_inches='tight', dpi=300, transparent=True)
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
        plt.bar(o, r2_o[o]*100, color='magenta', width=0.6)
    plt.xticks(list(r2_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks(list(r2_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
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
        plt.scatter(o + jitter, r2s[sort_by] * 100, c=colors, cmap='PRGn', s=20)
        plt.errorbar(o + 0.2, np.mean(r2s * 100), yerr=np.std(r2s * 100), color='k', alpha=0.5, fmt='o', capsize=0)
    plt.xticks(list(r2_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks(list(r2_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
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
        plt.bar(o, corr_z[o], color='magenta', width=0.6)
    plt.xticks(list(corr_z.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Correlation coefficient')
    plt.margins(0.1)
    plt.xticks(list(corr_z.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
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
    fig = plt.figure()
    sort_by = np.argsort(corr_o[0])[::-1]
    colors = np.linspace(0, 1, len(corr_o[0]))  # so colors are now in order of decreasing correlation scores for emission0
    for o in corr_o:
        coors = corr_o[o]
        jitter = np.random.uniform(-0.1, 0.1, len(coors))
        plt.scatter(o + jitter, coors[sort_by], c=colors, cmap='BrBG', s=20)
        plt.errorbar(o + 0.2, np.mean(coors), yerr=np.std(coors), color='k', alpha=0.5, fmt='o', capsize=0)

    plt.xticks(list(corr_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Correlation coefficient')
    plt.margins(0.1)
    plt.xticks(list(corr_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
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
        plt.bar(o, (lags_o[o] * 1000) / effective_fps, color='magenta', width=0.3)
    plt.xticks(list(lags_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Lag for max correlation coefficient (ms)')
    plt.margins(0.1)
    plt.xticks(list(lags_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
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
        plt.scatter(o + jitter, lags[sort_by], c=colors, cmap='BrBG', s=20)
        plt.errorbar(o + 0.2, np.mean(lags), yerr=np.std(lags), color='k', alpha=0.5, fmt='o', capsize=0)

    plt.xticks(list(lags_o.keys()), list(o_labels.values()), rotation=0)
    plt.ylabel('Lag for max correlation coefficient (ms)')
    plt.margins(0.1)
    plt.xticks(list(lags_o.keys()))
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
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
    fig, ax = plt.subplots(1, len(r2_z_o), figsize=(15, 7), sharey=True, layout='constrained')
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
    plt.suptitle(title)
    # plt.ylim(-10, 20)
    # plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    return fig


def plot_pearson_by_z_o(pearson_z_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot pearsonr in each state for each emission dimension separately
    """
    fig, ax = plt.subplots(1, len(pearson_z_o), figsize=(15, 7), sharey=True, layout='constrained')
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
    plt.suptitle(title)
    # plt.ylim(-10, 20)
    # plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_pearson_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    return fig


def plot_auxem_acc_by_z_o(acc_z_o, ay_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot accuracy scores in each state for each auxiliary emission dimension separately
    """
    fig, ax = plt.subplots(1, len(acc_z_o), figsize=(7, 4), sharey=True, layout='constrained')
    for z in acc_z_o:
        axes = ax[z] if len(acc_z_o) > 1 else ax
        for o in acc_z_o[z]:
            axes.bar(o, acc_z_o[z][o] * 100, color=COLORS[z])
        axes.set_xticks(list(acc_z_o[z].keys()), list(ay_labels.values()), rotation=0)
        axes.set_title(f'State {z+1}', color=COLORS[z])
        axes.axhline(0, c='k', ls=':', lw=2)
        axes.margins(0.1)

    axes = ax[0] if len(acc_z_o) > 1 else ax
    axes.set_ylabel('Balanced accuracy (%)')
    # plt.suptitle(title)
    plt.ylim(0, 100)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_auxem_acc_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_auxem_fraction_by_z_o(acc_z_o, ay_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot aux em event fractions in each state for each auxiliary emission dimension separately
    """
    fig, ax = plt.subplots(1, len(acc_z_o), figsize=(7, 4), sharey=True, layout='constrained')
    for z in acc_z_o:
        axes = ax[z] if len(acc_z_o) > 1 else ax
        for o in acc_z_o[z]:
            axes.bar(o, acc_z_o[z][o] * 100, color=COLORS[z])
        axes.set_xticks(list(acc_z_o[z].keys()), list(ay_labels.values()), rotation=0)
        axes.set_title(f'State {z+1}', color=COLORS[z])
        axes.axhline(0, c='k', ls=':', lw=2)
        axes.margins(0.1)

    axes = ax[0] if len(acc_z_o) > 1 else ax
    axes.set_ylabel('Fraction of behavior (%)')
    # plt.suptitle(title)
    plt.ylim(0, 100)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_auxem_fraction_by_z_and_o.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_var_explained_by_z_o_by_fly(r2_z_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot r2 scores in each state for each emission dimension separately, by fly
    """
    # print("o_labels", o_labels)
    fig, ax = plt.subplots(1, len(r2_z_o), figsize=(15, 7), sharey=True, layout='constrained')

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
            axes.scatter(o + jitter, scores[sort_by], c=colors, cmap=transparent_cmap, s=20)
            axes.errorbar(o + 0.2, np.mean(scores), yerr=np.std(scores), color='k', alpha=0.5, fmt='o', capsize=0)

        axes.set_xticks(list(r2_z_o[z].keys()), list(o_labels.values()), rotation=0)
        axes.set_title(f'State {z+1}', color=COLORS[z])
        axes.axhline(0, c='k', ls=':', lw=2)
        axes.margins(0.1)

    axes = ax[0] if len(r2_z_o) > 1 else ax
    axes.set_ylabel('Var explained (%)')
    plt.suptitle(title)
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
    fig, ax = plt.subplots(1, len(pearson_z_o), figsize=(15, 7), sharey=True, layout='constrained')

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
            axes.scatter(o + jitter, scores[sort_by], c=colors, cmap=transparent_cmap, s=20)
            axes.errorbar(o + 0.2, np.mean(scores), yerr=np.std(scores), color='k', alpha=0.5, fmt='o', capsize=0)

        axes.set_xticks(list(pearson_z_o[z].keys()), list(o_labels.values()), rotation=0)
        axes.set_title(f'State {z+1}', color=COLORS[z])
        axes.axhline(0, c='k', ls=':', lw=2)
        axes.margins(0.1)

    axes = ax[0] if len(pearson_z_o) > 1 else ax
    axes.set_ylabel('Pearson correlation coefficient (r)')
    plt.suptitle(title)
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


def plot_smoothed_probs(state_probs, model_config, data_config, batch, effective_fps, xlim=None, xlim_orig=None, prefix='', suffix='', savefig=False, fig_path=None, display=True):
    xlim_ = np.r_[xlim[0]:xlim[1]+1]

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    for z in range(model_config['num_states']):
        plt.plot(xlim_, state_probs[batch][xlim_, z], c=COLORS[z], linewidth=3, label=f'State {z+1}')

    plt.ylim([-0.05, 1.05])
    plt.yticks([0, 0.5, 1])

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
    plt.legend(loc='upper right')

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


def plot_trajectories(model_ckp, model_config, data_config, batch, prefix='', suffix='', xlim=None, xlim_orig=None, savefig=False, fig_path=None, display=True):

    # num_states = model_config['num_states']
    model_label = model_ckp['prefix'].upper() #+ '_' + str(num_states)
    emission_labels = data_config['emission_labels']
    emission_labels_zscored = data_config['emission_labels_zscored']

    emissions = model_ckp[f'{prefix}_data'][f'{prefix}_soft_predictions'][batch]
    true_emissions = model_ckp[f'{prefix}_data'][f'{prefix}_emissions'][batch]
    stateseq = model_ckp[f'{prefix}_data'][f'{prefix}_stateseq'][batch]

    plot_hmm_data_whole_session_with_states(
        emissions, true_emissions, stateseq, data_config, model_label=model_label, y_labels=emission_labels_zscored, xlim=xlim, xlim_orig=xlim_orig,
        title=f'Predicted female trajectory ({prefix.capitalize()}:{batch}) {suffix}',
        savefig=savefig, fig_path=fig_path, display=display)
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
        title=f'Sensory inputs and predicted female trajectory ({prefix.capitalize()}:{batch}) {suffix}', savefig=savefig, fig_path=fig_path, display=display)
    return
