from typing import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

from plotting.plots import *
from utilities.io import *
from utilities.utils import update_labels, generate_figures_filters_given_2datasets


def plot_state_o_dists_reformatted_2datasets(emissions1_z, emissions2_z, o_labels, title=None, savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(1, len(o_labels), figsize=(16, 5))

    def reformat(f_name, dt):
        if f_name in ['fFV', 'mFV']:
            t = dt
            xlim = (-1.5, 4.5)
            ylim = (0, 2.5)
        elif f_name in ['fLV', 'mLV']:
            t = dt
            xlim = (-2.5, 2.5)
            ylim = (0, 4)
        elif f_name in ['fAV', 'mAV']:
            t = dt
            xlim = (-150, 150)
            ylim = (0, 0.03)
        else:
            raise Exception(f'Unsupported o feat: {f_name}.')
        return t, xlim, ylim

    for o, ol in enumerate(o_labels):
        ax = axes[o] if len(o_labels) > 1 else axes
        for z in list(emissions1_z.keys()):
            data_z1 = emissions1_z[z][:, o]
            data_z_reformatted1, xlim, ylim = reformat(ol, data_z1)
            np.random.seed(0)
            samples1 = np.random.choice(np.round(data_z_reformatted1, decimals=3), min(10000, len(data_z_reformatted1)), replace=False)
            sns.kdeplot(samples1, color=COLORS[z], ax=ax,
                        common_norm=True,
                        label=f'LEAP rig',
                        alpha=1,
                        cut=0,
                        linewidth=2,
                        bw_adjust=2,
                        )

            data_z2 = emissions2_z[z][:, o]
            data_z_reformatted2, xlim, ylim = reformat(ol, data_z2)
            np.random.seed(0)
            samples2 = np.random.choice(np.round(data_z_reformatted2, decimals=3), min(10000, len(data_z_reformatted2)), replace=False)
            sns.kdeplot(samples2, color=COLORS[z], ax=ax,
                        common_norm=True,
                        label=f'16mic rig',
                        alpha=1,
                        cut=0,
                        linewidth=2,
                        bw_adjust=2,
                        linestyle='--',
                        )

        # ax.axvline(0, lw=0.5, c='gray', ls=':', alpha=0.7)
        ax.set_xlabel(o_labels[ol], color=EC)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # ax.legend(loc='upper right')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_mean_outputs_by_o_dists_reformatted_2datasets.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_aux_dists_reformatted_2datasets(aux_z1, aux_z2, a_labels, effective_fps1, effective_fps2, exclude_a=[], title=None, savefig=False, fig_dir=None, display=True):

    def reformat(f_name, dt, effective_fps):
        print(f_name)
        axtitle = None
        if f_name in ['mFV', 'mFS', 'fFV', 'fFS']:
            t = dt * effective_fps
            cut = 0
            xlabel = '(mm/s)'
            ylabel = 'Density'
            xlim = (-0.25, 1.25)
            xticks = [0, 0.5, 1,]
        elif f_name in ['mLS', 'fLS']:
            t = dt * effective_fps
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
            print(f_name, np.unique(dt, return_counts=True))
            print(f_name, np.unique(np.round(dt, 2), return_counts=True))
            print(f_name, np.sum(dt), len(dt))
            t = np.sum(dt) / len(dt)
            cut = None
            xlabel = None
            ylabel = f'Fraction'
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
        print(a, al)
        if al in exclude_a:
            continue
        for z in list(aux_z1.keys()):
            data_z1 = aux_z1[z][:, a]
            data_z2 = aux_z2[z][:, a]
            data_z_reformatted1, cut, xlabel, ylabel, xlim, xticks, axtitle = reformat(al, data_z1, effective_fps1)
            data_z_reformatted2, cut, xlabel, ylabel, xlim, xticks, axtitle = reformat(al, data_z2, effective_fps2)

            if al in ['pulse_i', 'sine_i', 'tap2']:
                ax[axi].bar(z/2-0.2, data_z_reformatted1, color=COLORS[z], alpha=0.9, width=0.25)
                ax[axi].bar(z/2+0.2, data_z_reformatted2, color=COLORS[z], alpha=0.5, width=0.25)
            else:
                samples1 = np.random.choice(data_z_reformatted1, min(10000, len(data_z1)), replace=False)
                sns.kdeplot(samples1, color=COLORS[z], ax=ax[axi],
                            label=f'LEAP rig',
                            alpha=1,
                            cut=cut,
                            linewidth=2,
                            )
                samples2 = np.random.choice(data_z_reformatted2, min(10000, len(data_z2)), replace=False)
                sns.kdeplot(samples2, color=COLORS[z], ax=ax[axi],
                            label=f'16mic rig',
                            alpha=1,
                            cut=cut,
                            linewidth=2,
                            linestyle='--',
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
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_mean_aux_dists_exclude_a={len(exclude_a)}_reformatted_2datasets.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def data(model_ckp):
    all_emissions = [*model_ckp['train_data']['train_emissions'], *model_ckp['test_data']['test_emissions']]
    all_stateseq = [*model_ckp['train_data']['train_stateseq'], *model_ckp['test_data']['test_stateseq']]
    all_aux_data = [*model_ckp['train_data']['train_aux_data'], *model_ckp['test_data']['test_aux_data']]
    all_output_mn_std = [*model_ckp['train_data']['train_output_mn_std'], *model_ckp['test_data']['test_output_mn_std']]
    all_aux_mn_std = [*model_ckp['train_data']['train_aux_mn_std'], *model_ckp['test_data']['test_aux_mn_std']]
    return all_emissions, all_stateseq, all_output_mn_std, all_aux_data, all_aux_mn_std


def make_plots(savefig=False, fig_dir=None, display=True):

    leap_model_ckp, leap_data_config, leap_model_config = load_specific_path(leap_model_dir)
    new16mic_model_ckp, new16mic_data_config, new16mic_model_config = load_specific_path(new16mic_model_dir)

    leap_effective_fps = leap_data_config['effective_fps']
    new16mic_effective_fps = new16mic_data_config['effective_fps']
    leap_all_emissions, leap_all_stateseq, leap_all_output_mn_std, leap_all_aux_data, leap_all_aux_mn_std = data(leap_model_ckp)
    new16mic_all_emissions, new16mic_all_stateseq, new16mic_all_output_mn_std, new16mic_all_aux_data, new16mic_all_aux_mn_std = data(new16mic_model_ckp)

    leap_aux_z = get_aux_by_state(leap_all_aux_data, leap_all_stateseq, 1, leap_all_aux_mn_std, rescaled=True)
    new16mic_aux_z = get_aux_by_state(new16mic_all_aux_data, new16mic_all_stateseq, 1, new16mic_all_aux_mn_std, rescaled=True)

    leap_emissions_z = get_emissions_by_state(leap_all_emissions, leap_all_stateseq, 1, leap_all_output_mn_std, rescaled=True, effective_fps=leap_effective_fps)
    new16mic_emissions_z = get_emissions_by_state(new16mic_all_emissions, new16mic_all_stateseq, 1, new16mic_all_output_mn_std, rescaled=True, effective_fps=new16mic_effective_fps)

    update_labels(leap_data_config)

    # plots.plot_legends(1, leap_data_config, savefig=savefig, fig_dir=fig_dir, display=display)
    #
    # plots.plot_state_o_dists_reformatted_2datasets(
    #     leap_emissions_z, new16mic_emissions_z, leap_data_config['emission_labels_units'],
    #     title=f'all data', savefig=savefig, fig_dir=fig_dir, display=display
    # )

    plot_state_aux_dists_reformatted_2datasets(
        leap_aux_z, new16mic_aux_z, leap_data_config['auxiliary_labels'], leap_effective_fps, new16mic_effective_fps,
        exclude_a=['wingAlign', 'fmAng_sin'], title=f'all data', savefig=savefig, fig_dir=fig_dir, display=display
    )

    plot_state_aux_dists_reformatted_2datasets(
        leap_aux_z, new16mic_aux_z, leap_data_config['auxiliary_labels'], leap_effective_fps, new16mic_effective_fps,
        title=f'all data', savefig=savefig, fig_dir=fig_dir, display=display
    )
    return


def get_filter_amplitudes(weights, data_config, y_labels, input_mask_by_emission, skip_states=[]):
    # print(weights.shape)

    num_states = weights.shape[0]
    basis = data_config['basis']

    print("weights.shape", weights.shape)   # (num states, <emission_dim>, transformed filterlen), e.g. (5, 4, 84)
    # <emission_dim> is 4 here coz locomotion + wingflick

    w_amps = dict()

    for d, _ in enumerate(y_labels):
        e_mask = input_mask_by_emission[d]
        print("y_labels[_]", _, y_labels[_])
        weights_d = weights[:, [d]][..., e_mask == 1]
        print("weights_d pre basistr", weights_d.shape)     # (num states, 1, transformed filterlen), e.g. (5, 1, 28)
        weights_d = basis_invtransform_one_by_one(weights_d, basis, n_inputs=len(y_labels[_]))[:, 0]
        print("weights_d post basistr", weights_d.shape)    # (num states, n_inputs, orig filterlen), e.g. (5, 7, 450)
        w_amps[d] = []
        for z in range(num_states):
            if z in skip_states:
                continue
            w_l2 = np.linalg.norm(weights_d[z], axis=-1)
            w_amps[d].append(w_l2)
            print(f'State {z+1}', _, "w_l2.shape", w_l2.shape)
        w_amps[d] = np.array(w_amps[d])
        w_amps[d] = w_amps[d] / np.max(w_amps[d][1:])
    return w_amps


# def get_configs(model_dir):
#     model_ckp, data_config, model_config = load_specific_path(model_dir)
#     return model_ckp, data_config, model_config


def plot_weight_magnitudes(savefig=True, display=True):

    _, leap_data_config, leap_model_config = load_specific_path(leap_model_dir)
    _, new16mic_data_config, new16mic_model_config = load_specific_path(new16mic_model_dir)

    leap_input_mask_by_emission = leap_data_config['input_mask_by_emission']
    new16mic_input_mask_by_emission = new16mic_data_config['input_mask_by_emission']
    emission_labels = leap_data_config['emission_labels']
    update_labels(leap_data_config)
    input_labels = leap_data_config['input_labels']
    emission_labels_dict = leap_data_config['emission_labels_dict']
    print("emission_labels_dict", emission_labels_dict)

    avg_weight_WT = joblib.load(f'{leap_model_dir}/avg_weight_WT.pkl')
    avg_weight_WT_FRED =  joblib.load(f'{new16mic_model_dir}/avg_weight_WT_FRED.pkl')

    leap_w_amps = get_filter_amplitudes(avg_weight_WT, leap_data_config, emission_labels, leap_input_mask_by_emission, skip_states=[])
    new16mic_w_amps = get_filter_amplitudes(avg_weight_WT_FRED, new16mic_data_config, emission_labels, new16mic_input_mask_by_emission, skip_states=[])

    # print(leap_w_amps, leap_w_amps[0].shape)
    # print(new16mic_w_amps)

    num_states = avg_weight_WT.shape[0]
    fig, ax = plt.subplots(1, len(emission_labels), figsize=(12+3, 4.1))
    for d, _ in enumerate(emission_labels):
        limit = max(leap_w_amps[d][1:].max(), new16mic_w_amps[d][1:].max()) * 1.1
        # print("limit", limit, leap_w_amps[d][1:].max(), new16mic_w_amps[d][1:].max())
        ax[d].plot([0, limit], [0, limit], 'k--', alpha=0.3, zorder=0)
        for z in range(num_states):
            if z == 0: continue
            ax[d].scatter(leap_w_amps[d][z], new16mic_w_amps[d][z], color=COLORS[z], label=f'State {z+1}')
            print(f'State {z+1}', _, ': r2 score: ', r2_score(leap_w_amps[d][z], new16mic_w_amps[d][z]))
        ax[d].set_title(emission_labels_dict[_])
        ax[d].set_xticks([0, 0.5, 1])
        ax[d].set_yticks([0, 0.5, 1])
        ax[d].margins(0.1)
        print([input_labels[i] for i in emission_labels[_]])
        if ax[d].get_subplotspec().is_last_col():
            ax[d].legend(loc='upper right', bbox_to_anchor=(2, 1), borderaxespad=0.)
        ax[d].set_xlabel('Norm. |w|\n(Dataset 1)')
        ax[d].set_ylabel('Norm. |w|\n(Dataset 2)')
    return
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, 'weightcorr_2datasets.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_weights():
    avg_weight_WT = joblib.load(f'{leap_model_dir}/avg_weight_WT.pkl')
    avg_weight_WT_FRED = joblib.load(f'{new16mic_model_dir}/avg_weight_WT_FRED.pkl')

    _, leap_data_config, leap_model_config = load_specific_path(leap_model_dir)
    _, new16mic_data_config, new16mic_model_config = load_specific_path(new16mic_model_dir)

    generate_figures_filters_given_2datasets(leap_data_config, new16mic_data_config, avg_weight_WT, avg_weight_WT_FRED, fig_dir, savefig=True, display=False)
    return


def get_emp_occ(state_seqs, config):
    from collections import defaultdict
    ps_z = defaultdict(list)
    for i in range(len(state_seqs)):
        state_z, count_z = np.unique(state_seqs[i], return_counts=True)
        # print(i, state_z, count_z)
        percent_z = count_z / np.sum(count_z)
        s_p_dict = dict(zip(state_z, percent_z))
        for z in range(config['num_states']):
            ps_z[z].append(s_p_dict.get(z, 0))
    return ps_z


def plot_empirical_occupancy_2datasets(savefig=True, display=True):

    leap_model_ckp, leap_data_config, leap_model_config = load_specific_path(leap_model_dir)
    new16mic_model_ckp, new16mic_data_config, new16mic_model_config = load_specific_path(new16mic_model_dir)

    leap_z_seqs = [*leap_model_ckp['train_data']['train_stateseq'], *leap_model_ckp['test_data']['test_stateseq']]
    new16mic_z_seqs = [*new16mic_model_ckp['train_data']['train_stateseq'], *new16mic_model_ckp['test_data']['test_stateseq']]

    leap_ps_z = get_emp_occ(leap_z_seqs, leap_model_config)
    new16mic_ps_z = get_emp_occ(new16mic_z_seqs, new16mic_model_config)

    pmax = -1
    fig = plt.figure(figsize=(6, 4))
    for z in leap_ps_z:
        jitter1 = np.random.uniform(-0.1, 0.1, len(leap_ps_z[z]))
        jitter2 = np.random.uniform(-0.1, 0.1, len(new16mic_ps_z[z]))
        plt.scatter(z+1-0.2-jitter1, leap_ps_z[z], c=COLORS[z], s=SCATTERSIZE, edgecolors='none', label=f'Dataset 1' if z == 0 else None)
        plt.scatter(z+1+0.2+jitter2, new16mic_ps_z[z], c=COLORS[z], alpha=0.5, s=SCATTERSIZE, edgecolors='none', label=f'Dataset 2' if z == 0 else None)
        plt.errorbar(z+1-0.04, np.mean(leap_ps_z[z]), yerr=np.std(leap_ps_z[z]), color='k', alpha=0.8, fmt='o', capsize=0)
        plt.errorbar(z+1+0.36, np.mean(new16mic_ps_z[z]), yerr=np.std(new16mic_ps_z[z]), color='k', alpha=0.8, fmt='o', capsize=0)
        pmax = max(pmax, max(np.max(new16mic_ps_z[z]), np.max(leap_ps_z[z])))

    plt.ylabel('Fraction occupancy')
    plt.legend(loc='upper left')
    plt.margins(0.1)
    pmax = np.round(pmax, 1) + 0.2
    plt.ylim(-0.1, pmax)
    plt.yticks([0, pmax / 3, 2*pmax/3, pmax])
    plt.xticks(range(1, 1 + len(leap_ps_z)))
    plt.xlabel('State')
    if savefig: fig.savefig(os.path.join(fig_dir, f'empirical_occupancy_2datasets.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return fig


if __name__ == '__main__':
    leap_model_dir = '../paper figs/FINAL WT/20260101_235805_duration/'
    new16mic_model_dir = '../paper figs/FINAL WT FRED/20260102_135949_spandex/'

    fig_dir = os.path.join('../paper figs/figure comparison2datasets/')
    # make_plots(savefig=True, fig_dir=fig_dir, display=True)
    # plot_empirical_occupancy_2datasets()
    plot_weight_magnitudes()
    # plot_weights()
    # plot_legends(0, 0, savefig=True, fig_dir=fig_dir, display=False)
