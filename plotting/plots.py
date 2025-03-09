import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
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
mpl.rcParams['axes.linewidth'] = 2
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

# -- Fonts --
mpl.rcParams['font.size'] = 16  # Panel label
# mpl.rcParams['font.family'] = 'Arial'
# mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'

# -- Figure size --
# plt.rcParams['figure.figsize'] = (6, 4)
# plt.rcParams['figure.dpi'] = 300

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


def plot_hmm_data_whole_session_with_states(predicted_emissions, true_emissions, predicted_states, model_label=None, xlim=None, y_labels=None, title=None):
    """Plot emissions vs. time"""
    num_timesteps = len(predicted_emissions)
    emission_dim = predicted_emissions.shape[-1]
    fig, axs = plt.subplots(emission_dim, 1, figsize=(15, 10), sharex=True)
    d = 0
    for _ in y_labels:
        ax = axs[d]

        max_value = max(abs(predicted_emissions[:, d]).max(), abs(true_emissions[:, d]).max())
        lim = 1.05 * max_value
        ax.imshow(predicted_states[None, :], aspect="auto", interpolation="none", cmap=CMAP, vmin=0, vmax=len(COLORS)-1,
                  extent=(0, num_timesteps, -lim, lim))

        ax.plot(true_emissions[:, d], "k-", alpha=0.6, label='Data')
        ax.plot(predicted_emissions[:, d], 'm-', linewidth=1.5, label=f'{model_label}')
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


def plot_filters(weights, data_config, savefig=False, fig_dir=None, display=True):
    print(weights.shape)

    num_states = weights.shape[0]
    emission_dim = weights.shape[1]
    filter_len = weights.shape[-1]
    emission_labels = data_config['emission_labels']
    input_labels = data_config['input_labels']
    n_inputs = len(input_labels)
    basis = data_config['basis']

    if basis is not None:
        weights = BasisProjection(basis).inverse_transform(weights.reshape(-1, filter_len)).reshape(num_states, emission_dim, -1)
    weights = weights.reshape(num_states, emission_dim, n_inputs, -1)

    fig, axs = plt.subplots(emission_dim, n_inputs, figsize=(20, 10), sharex=True, sharey='row')
    # weights = weights / np.sum(weights)
    d = 0
    for _ in emission_labels:
        w = weights[:, d]
        w = w/np.sum(w)
        stim = 0
        for __ in input_labels:
            ax = axs[d, stim] if emission_dim > 1 else axs[stim]
            for s in range(num_states):
                ax.plot(np.arange(-w[s, stim].shape[-1], 0)/150, w[s, stim], color=COLORS[s], linewidth=2.1)
            if d == 0:
                ax.set_title(f'{input_labels[__]}', fontsize='medium')
            ax.axhline(0, ls=':', c='k')
            stim += 1
            ax.margins(0.05)
        ax = axs[d, 0] if emission_dim > 1 else axs[0]
        ax.set_ylabel(emission_labels[_], fontsize='medium', color='magenta')
        d += 1
    # if num_states > 1:
    #     ax = axs[0, 0] if emission_dim > 1 else axs[0]
    #     ax.legend([f'State {s}' for s in range(num_states)], loc='upper left', fontsize='x-small')
    fig.supylabel('Filter amplitude (a.u.)')
    fig.supxlabel("Time relative to prediction (s)")
    fig.align_ylabels(axs[:, 0])
    plt.margins(0.02)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, 'filters.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_state_mean_inputs(model_config, inputs_z, data_config):
    input_labels = data_config['input_labels']
    n_inputs = len(input_labels)
    basis = data_config['basis']
    num_states = model_config['num_states']

    fig, ax = plt.subplots(1, n_inputs, sharex=True)
    for z in range(num_states):
        inputs_mean = np.mean(inputs_z[z], axis=0)
        # outputs_mean, outputs_var = np.mean(outputs_z[z], axis=1), np.cov(outputs_z[z])
        print(z, inputs_mean.shape)
        if basis is not None:
            inputs_mean = BasisProjection(basis).inverse_transform(inputs_mean)
        print(z, inputs_mean.shape)
        inputs_mean = inputs_mean.reshape(n_inputs, -1)
        stim = 0
        for __ in input_labels:
            # ax[stim].plot(np.arange(-inputs_mean[stim].shape[-1], 0)/150, inputs_mean[stim][:], c=COLORS[z], linewidth=3, label=f'State {z}')
            # ax[stim].scatter(stim, np.mean(inputs_mean[stim]), c=COLORS[z], label=f'State {z}')
            ax[stim].errorbar(stim, np.mean(inputs_mean[stim]), yerr=np.std(inputs_mean[stim]), fmt='o', c=COLORS[z], label=f'State {z}')
            ax[stim].set_title(input_labels[__])
            ax[stim].axhline(0, ls=':', c='k')
            if stim == 0:
                ax[stim].legend(loc='upper left')
            ax[stim].margins(0.2)
            ax[stim].set_xticklabels([])
            ax[stim].set_xticks([])
            stim += 1
            # ax.plot(np.arange(-w[s, stim].shape[-1], 0) / 150, w[s, stim], color=COLORS[s], linewidth=2.1)
    # fig.supxlabel("Time relative to prediction (s)")
    plt.suptitle('Average sensory inputs (zscored where applicable)')
    plt.tight_layout()
    return fig


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


def plot_state_mean_outputs_by_o_dists(model_config, outputs_z, data_config):

    emission_labels = data_config['emission_labels']
    num_states = model_config['num_states']

    fig, ax = plt.subplots(2, len(emission_labels))

    o = 0
    for __ in emission_labels:
        for z in range(num_states):
            rand_idxs = np.random.randint(len(outputs_z[z][:, o]), size=1000)
            ax[0, o].scatter(
                np.random.uniform(z-0.1, z+0.1, len(rand_idxs)),
                outputs_z[z][rand_idxs, o],
                c=COLORS[z],
                s=2,
            )
        ax[0, o].set_ylabel(f'z-scored {emission_labels[__]} (a.u.)', color='magenta')
        ax[0, o].set_xticks(range(num_states))
        ax[0, o].set_xlabel('State')
        ax[0, o].margins(0.1)
        ax[0, o].axhline(0, ls=":", lw=2, c='k')
        o += 1

    o = 0
    for __ in emission_labels:
        for z in range(num_states):
            data = np.round(outputs_z[z][:, o], decimals=10)
            # rand_idxs = np.random.randint(len(outputs_z[z][:, o]), size=100)
            print(o, z, np.std(data))
            sns.histplot(data, color=COLORS[z], ax=ax[1, o],
                        common_norm=False,
                        log_scale=True,
                         kde=True,
                         stat='density',
                        label=f'State {z}',
                         edgecolor=None,
                         alpha=1,
                         bins=100,
                        )
        ax[1, o].set_xlabel(f'z-scored {emission_labels[__]} (a.u.)', color='magenta')
        ax[1, o].margins(y=0.2)
        ax[1, o].axhline(0, ls=":", lw=2, c='k')
        if o == 0:
            ax[1, 0].legend(loc='upper left')
        o += 1
    plt.suptitle('Distribution of female outputs by state')
    plt.tight_layout()
    return fig


def plot_prob_states(state_seqs, config, title=None, savefig=False, fig_dir=None, display=True):
    print("state_seqs", state_seqs.shape)

    fig = plt.figure()
    for z in range(config['num_states']):
        prob_z = np.mean(state_seqs == z, axis=0)  # Probability of z at each time step
        plt.plot(uniform_filter1d(prob_z, size=100), c=COLORS[z], linewidth=1.5, label=f'State {z}')
    plt.xlabel('Time (min)', fontsize='large')
    plt.legend(loc='upper right')
    plt.margins(0.05)
    plt.ylabel('P(state)', fontsize='large')
    plt.xticks([0, len(prob_z)], [0, 15], fontsize='medium')
    plt.yticks(fontsize='medium')
    plt.title(f'State occupancy over all {title} sessions')
    plt.ylim(0, 1)
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_prob_states_over_time.pdf'), bbox_inches='tight', dpi=300,
                            transparent=True)
    if display: plt.show()
    plt.close()
    return fig


def plot_transition_matrix(transition_matrix, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(20, 20))
    m = transition_matrix.shape[0]
    sns.heatmap(transition_matrix, annot=True, cmap='bone', cbar=True, square=True, fmt=".3f",
                xticklabels=[f'State {i}' for i in range(m)],
                yticklabels=[f'State {i}' for i in range(m)])
    plt.title('Transition Matrix')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, 'transition_matrix.pdf'), bbox_inches='tight', dpi=300)
    if display: plt.show()
    plt.close()
    return fig


def plot_ethogram(transition_matrix, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(10, 10))

    G = nx.DiGraph()
    num_states = transition_matrix.shape[0]

    # Add edges with weights
    for i in range(num_states):
        for j in range(num_states):
            if transition_matrix[i, j] > 0.005:  # Only add edges with nonzero probability
                G.add_edge(i, j, weight=transition_matrix[i, j])

    pos = nx.spring_layout(G)

    edges = G.edges(data=True)
    edge_widths = [d['weight'] * 5 for (u, v, d) in edges]  # Scale edge width
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000,
            font_size=14, font_weight='bold', edge_color='gray', width=edge_widths,
            arrows=True,
            connectionstyle='arc3,rad=0.1')

    # Draw edge labels
    edge_labels = {(u, v): f"{d['weight']:.3f}" for (u, v, d) in edges}
    # print(edge_labels, len(edge_labels))
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.4, connectionstyle='arc3,rad=0.1')

    plt.title("Transition Probability Graph")
    if savefig: fig.savefig(os.path.join(fig_dir, 'ethogram.pdf'), bbox_inches='tight', dpi=300)
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
    print("communities", communities)

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
    if savefig: fig.savefig(os.path.join(fig_dir, f'ethogram_community_{threshold}.pdf'), bbox_inches='tight', dpi=300)
    if display: plt.show()
    plt.close()
    return


def plot_steady_state(steady_state_p, savefig=False, fig_dir=None, display=True):
    fig = plt.figure()
    # m = steady_state_p.shape[0]
    # sns.heatmap(steady_state_p[None, :], annot=True, cmap='bone', cbar=False, square=True,
    #             fmt=".2f",
    #             xticklabels=[f'State {_}' for _ in range(m)])
    # plt.yticks([])
    # plt.title('Steady State Probabilities')

    for z in range(len(steady_state_p)):
        plt.bar(z, steady_state_p[z] * 100, color=COLORS[z])
    plt.ylabel('Time spent\nin State(%)')
    plt.margins(0.1)
    plt.xticks(range(len(steady_state_p)))
    plt.xlabel('State')
    plt.title('State Occupancy')
    if savefig: fig.savefig(os.path.join(fig_dir, 'steady_state_p.pdf'), bbox_inches='tight', dpi=300)
    if display: plt.show()
    plt.close()
    return fig


def plot_var_explained(r2, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot overall r2 scores.
    :return:
    """
    fig = plt.figure()
    plt.bar(1, r2 * 100)
    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks()
    plt.xlabel('')
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score.pdf'), bbox_inches='tight', dpi=300)
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
        plt.bar(z, r2_z[z] * 100, color=COLORS[z])
    plt.ylabel('Var explained (%)')
    plt.margins(0.1)
    plt.xticks(list(r2_z.keys()))
    plt.xlabel('State')
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    # plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_z.pdf'), bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return fig


def plot_correlation_by_o(corr_z, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot correlation coefficients for each emission timeseries.
    :return:
    """
    fig = plt.figure()
    for z in corr_z:
        plt.bar(z, corr_z[z], color='magenta')
    plt.xticks(list(corr_z.keys()), list(o_labels.values()), rotation=90)
    plt.ylabel('Correlation (lag=0)')
    plt.margins(0.1)
    plt.xticks(list(corr_z.keys()))
    plt.xlabel('State')
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    # plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_correlation_by_o.pdf'), bbox_inches='tight', dpi=300)
    if display: plt.show()
    return fig


def plot_var_explained_by_z_o(r2_z_o, o_labels, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot r2 scores in each state for each emission dimension separately
    """
    print("o_labels", o_labels)
    fig, ax = plt.subplots(1, len(r2_z_o), figsize=(10, 10), sharey=True, layout='constrained')
    for z in r2_z_o:
        for o in r2_z_o[z]:
            ax[z].bar(o, r2_z_o[z][o] * 100, color=COLORS[z])
        ax[z].set_xticks(list(r2_z_o[z].keys()), list(o_labels.values()), rotation=90)
        ax[z].set_title(f'State {z}', color=COLORS[z])
        ax[z].axhline(0, c='k', ls=':', lw=2)
        ax[z].margins(0.1)
    ax[0].set_ylabel('Var explained (%)')
    plt.suptitle(title)
    # plt.ylim(-10, 20)
    # plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_score_by_z_and_o.pdf'), bbox_inches='tight', dpi=300)
    if display: plt.show()
    return fig


def plot_loss(em_losses, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    plt.plot(em_losses, '.-')
    print("em_losses:", em_losses)
    plt.title('EM training iters')
    plt.xlabel('#iters')
    plt.ylabel('Neg Log prob (per frame)')
    plt.margins(0.2)
    if savefig: fig.savefig(os.path.join(fig_dir, 'loss.pdf'), bbox_inches='tight', dpi=300)
    if display: plt.show()
    return fig


# def print_params(params, true_params=False):
#     jnp.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#     if true_params:
#         prefix = 'True'
#     else:
#         prefix = 'Fitted'
#     print(f"{prefix} initial probs:")
#     print(params.initial.probs)
#     print(f"{prefix} transition matrix:")
#     print(params.transitions.transition_matrix)
#     print(f"{prefix} emission input weights:")
#     print(params.emissions.weights)
#     print(f"{prefix} emission input biases:")
#     print(params.emissions.biases)
#     print(f"{prefix} emission input covs:")
#     print(params.emissions.covs)


# def print_ghmm_params(params, true_params=False):
#     jnp.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#     if true_params:
#         prefix = 'True'
#     else:
#         prefix = 'Fitted'
#     print(f"{prefix} initial probs:")
#     print(params.initial.probs)
#     print(f"{prefix} transition matrix:")
#     print(params.transitions.transition_matrix)
#     print(f"{prefix} emissions:")
#     print(params.emissions)


# def plot_trajectories(model_ckp, data_config, prefix_data='', xlim=None, savefig=False, fig_dir=None, display=True):
#     """
#
#     :param prefix_data: 'train' or 'test'
#     :return:
#     """
#
#     data_key = f'{prefix_data}_data'
#     emissions_key = f'{prefix_data}_emissions'
#     predictions_key = f'{prefix_data}_predictions'
#     stateseq_key = f'{prefix_data}_stateseq'
#
#     num_states = model_ckp['model'].num_states
#     model_label = model_ckp['prefix'].upper() + '_' + str(num_states)
#     emission_labels = data_config['emission_labels']
#
#     sessions = np.random.choice(range(len(model_ckp[data_key][emissions_key])), 3, replace=False)   # plotting random 3 sessions
#     os.makedirs(f'{fig_dir}/trajs', exist_ok=True)
#     for btch in sessions:
#         emissions = model_ckp[data_key][predictions_key][btch]
#         true_emissions = model_ckp[data_key][emissions_key][btch]
#         stateseq = model_ckp[data_key][stateseq_key][btch]
#
#         plot_hmm_data_whole_session_with_states(
#             emissions, true_emissions, stateseq, model_label=model_label, y_labels=emission_labels,
#             title=f'Predicted female trajectory ({prefix_data.capitalize()}:{btch})')
#
#         if savefig: plt.savefig(f'{fig_dir}/trajs/{prefix_data}_{btch}.pdf', dpi=300, bbox_inches='tight', transparent=True)
#         if display: plt.show()
#         plt.close()
#
#         for xlim in [(0, 1000), (10000, 11000), (15000, 16000), (0, 5000), (0, 200), (1500, 1600)]:
#             plot_hmm_data_whole_session_with_states(
#                 emissions, true_emissions, stateseq, model_label=model_label, y_labels=emission_labels, xlim=xlim,
#                 title=f'Predicted female trajectory ({prefix_data.capitalize()}:{btch})')
#             if savefig: plt.savefig(f'{fig_dir}/trajs/{prefix_data}_{btch}_xlim={xlim}.pdf', dpi=300,
#                         bbox_inches='tight', transparent=True)
#             if display: plt.show()
#             plt.close()
#     return


def plot_smoothed_probs(state_probs, batch, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(20, 5))
    plt.plot(uniform_filter1d(state_probs[batch], size=500, axis=0))
    plt.ylim([0, 1])
    if savefig: fig.savefig(os.path.join(fig_dir, f"smoothed_probs_{batch}.pdf"), dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_trajectories(model_ckp, model_config, data_config, batch, prefix_data='', xlim=None, savefig=False, fig_path=None, display=True):
    """

    :param prefix_data: 'train' or 'test'
    :param btch
    :return:
    """

    data_key = f'{prefix_data}_data'
    emissions_key = f'{prefix_data}_emissions'
    predictions_key = f'{prefix_data}_predictions'
    stateseq_key = f'{prefix_data}_stateseq'

    num_states = model_config['num_states']
    model_label = model_ckp['prefix'].upper() + '_' + str(num_states)
    emission_labels = data_config['emission_labels']

    sessions = [batch]   # plot that session

    for b in sessions:
        emissions = model_ckp[data_key][predictions_key][b]
        true_emissions = model_ckp[data_key][emissions_key][b]
        stateseq = model_ckp[data_key][stateseq_key][b]

        # fig_path = os.path.join(fig_dir, f'{prefix_data}_{b}_xlim={xlim}.pdf')
        fig = plot_hmm_data_whole_session_with_states(
            emissions, true_emissions, stateseq, model_label=model_label, y_labels=emission_labels, xlim=xlim,
            title=f'Predicted female trajectory ({prefix_data.capitalize()}:{b})')

        if savefig: plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
        if display: plt.show()
        plt.close()
    return
