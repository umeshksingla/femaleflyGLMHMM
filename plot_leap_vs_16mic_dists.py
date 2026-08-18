import seaborn as sns
from seaborn import xkcd_rgb
from sklearn.metrics import r2_score

from utilities.io import *
from preprocess.colors import EC, colors_datasource
from utilities.utils import update_labels, generate_figures_filters_given_2datasets


def plot_legends(ds1, ds2, savefig=False, fig_dir=None, display=True):

    fig, ax = plt.subplots(figsize=(2.4, 1.2))
    ax.plot([0, 1], [1, 1], c=xkcd_rgb[colors_datasource[ds1]], linewidth=6)
    ax.text(1.2, 1, ds1, va='center', fontsize='small')
    ax.plot([0, 1], [0, 0], c=xkcd_rgb[colors_datasource[ds2]], linewidth=6, alpha=0.5, linestyle='--')
    ax.text(1.2, 0, ds2, va='center', fontsize='small')
    # ax.set_ylim([0.5, -num_states-0.5])
    ax.axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{fig_dir}/{ds1}-{ds2}_legend.pdf', dpi=300, transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()
    return


def plot_o_dists_reformatted_2datasets(ds1, ds2, emissions1, emissions2, o_labels, title=None, savefig=False, fig_dir=None, display=True):

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
        data1 = emissions1[:, o]
        data_reformatted1, xlim, ylim = reformat(ol, data1)
        np.random.seed(0)
        samples1 = np.random.choice(np.round(data_reformatted1, decimals=3), min(10000, len(data_reformatted1)), replace=False)
        sns.kdeplot(samples1, color=xkcd_rgb[colors_datasource[ds1]], ax=ax,
                    common_norm=True,
                    label=ds1,
                    alpha=1,
                    cut=0,
                    linewidth=2,
                    bw_adjust=2,
                    )

        data2 = emissions2[:, o]
        data_reformatted2, xlim, ylim = reformat(ol, data2)
        np.random.seed(0)
        samples2 = np.random.choice(np.round(data_reformatted2, decimals=3), min(10000, len(data_reformatted2)), replace=False)
        sns.kdeplot(samples2, color=xkcd_rgb[colors_datasource[ds2]], ax=ax,
                    common_norm=True,
                    label=ds2,
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

    axes[0].legend(loc='upper right')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_mean_outputs_by_o_dists_reformatted_2datasets.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_aux_dists_reformatted_2datasets(ds1, ds2, aux1, aux2, a_labels, effective_fps1, effective_fps2, exclude_a=[], title=None, savefig=False, fig_dir=None, display=True):

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
        data_z1 = aux1[:, a]
        data_z2 = aux2[:, a]
        data_z_reformatted1, cut, xlabel, ylabel, xlim, xticks, axtitle = reformat(al, data_z1, effective_fps1)
        data_z_reformatted2, cut, xlabel, ylabel, xlim, xticks, axtitle = reformat(al, data_z2, effective_fps2)

        if al in ['pulse_i', 'sine_i', 'tap2']:
            ax[axi].bar(-0.2, data_z_reformatted1, color=xkcd_rgb[colors_datasource[ds1]], alpha=0.9, width=0.25)
            ax[axi].bar(+0.2, data_z_reformatted2, color=xkcd_rgb[colors_datasource[ds2]], alpha=0.5, width=0.25)
        else:
            samples1 = np.random.choice(data_z_reformatted1, min(10000, len(data_z1)), replace=False)
            sns.kdeplot(samples1, color=xkcd_rgb[colors_datasource[ds1]], ax=ax[axi],
                        label=ds1,
                        alpha=1,
                        cut=cut,
                        linewidth=2,
                        )
            samples2 = np.random.choice(data_z_reformatted2, min(10000, len(data_z2)), replace=False)
            sns.kdeplot(samples2, color=xkcd_rgb[colors_datasource[ds2]], ax=ax[axi],
                        label=ds2,
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
    # ax[0].legend(loc='upper right')
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


def make_plots(ds1='', ds2='', savefig=False, fig_dir=None, display=True):
    ds1_model_dir = model_paths[ds1]
    ds2_model_dir = model_paths[ds2]

    ds1_model_ckp, ds1_data_config, ds1_model_config = load_specific_path(ds1_model_dir)
    ds2_model_ckp, ds2_data_config, ds2_model_config = load_specific_path(ds2_model_dir)

    ds1_effective_fps = ds1_data_config['effective_fps']
    ds2_effective_fps = ds2_data_config['effective_fps']
    ds1_all_emissions, ds1_all_stateseq, ds1_all_output_mn_std, ds1_all_aux_data, ds1_all_aux_mn_std = data(ds1_model_ckp)
    ds2_all_emissions, ds2_all_stateseq, ds2_all_output_mn_std, ds2_all_aux_data, ds2_all_aux_mn_std = data(ds2_model_ckp)

    ds1_all_stateseq = [np.zeros(_.shape) for _ in ds1_all_stateseq]    # all states are just 0s for these plots
    ds2_all_stateseq = [np.zeros(_.shape) for _ in ds2_all_stateseq]

    ds1_aux = get_aux_by_state(ds1_all_aux_data, ds1_all_stateseq, 1, ds1_all_aux_mn_std, rescaled=True)
    ds2_aux = get_aux_by_state(ds2_all_aux_data, ds2_all_stateseq, 1, ds2_all_aux_mn_std, rescaled=True)

    ds1_emissions = get_emissions_by_state(ds1_all_emissions, ds1_all_stateseq, 1, ds1_all_output_mn_std, rescaled=True, effective_fps=ds1_effective_fps)
    ds2_emissions = get_emissions_by_state(ds2_all_emissions, ds2_all_stateseq, 1, ds2_all_output_mn_std, rescaled=True, effective_fps=ds2_effective_fps)
    update_labels(ds1_data_config)

    ds1_aux = ds1_aux[0]            # everything is in just state 0
    ds2_aux = ds2_aux[0]
    ds1_emissions = ds1_emissions[0]
    ds2_emissions = ds2_emissions[0]

    prefix = f'{ds1}-{ds2}'

    plot_legends(ds1, ds2, savefig=savefig, fig_dir=fig_dir, display=display)

    plot_o_dists_reformatted_2datasets(
        ds1, ds2, ds1_emissions, ds2_emissions, ds1_data_config['emission_labels_units'],
        title=prefix, savefig=savefig, fig_dir=fig_dir, display=display
    )

    plot_aux_dists_reformatted_2datasets(
        ds1, ds2, ds1_aux, ds2_aux, ds1_data_config['auxiliary_labels'], ds1_effective_fps, ds2_effective_fps,
        exclude_a=['wingAlign', 'fmAng_sin'], title=prefix, savefig=savefig, fig_dir=fig_dir, display=display
    )

    plot_aux_dists_reformatted_2datasets(
        ds1, ds2, ds1_aux, ds2_aux, ds1_data_config['auxiliary_labels'], ds1_effective_fps, ds2_effective_fps,
        title=prefix, savefig=savefig, fig_dir=fig_dir, display=display
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

    # avg_weight_WT = joblib.load(f'{leap_model_dir}/avg_weight_wt.pkl')
    # avg_weight_WT_FRED =  joblib.load(f'{new16mic_model_dir}/avg_weight_wt_fred.pkl')

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
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, 'weightcorr_2datasets.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_weights():
    # all_weights_wt = joblib.load(f'{leap_model_dir}/all_weights_wt.pkl')
    # all_weights_wt_fred = joblib.load(f'{new16mic_model_dir}/all_weights_wt_fred.pkl')

    # avg_weight_WT = np.mean(all_weights_wt, axis=0)
    # avg_weight_WT_FRED = np.mean(all_weights_wt_fred, axis=0)

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

    model_paths = {
        'wt': '../FINAL_WT/20260531_183347_obsidian/',       # wt female
        'wt_16mic': '../FINAL_WT_FRED/20260626_050049_whelp',    # wt female fred
        'ac_both': 'models/general_ac_both/idarhmm_5_cv/20260814_173744_accounting/',         # ac both female
        'blind_both': 'models/general_blind_both/idarhmm_5_cv/20260814_175044_congregation/',  # blind both female
    }

    # ds1_model_dir = model_paths['wt']
    # ds2_model_dir = model_paths['ac_both']

    # all_weights_wt = joblib.load(f'{leap_model_dir}/all_weights_wt.pkl')
    # all_weights_wt_fred = joblib.load(f'{new16mic_model_dir}/all_weights_wt_fred.pkl')
    #
    # avg_weight_WT = np.mean(all_weights_wt, axis=0)
    # avg_weight_WT_FRED = np.mean(all_weights_wt_fred, axis=0)

    fig_dir = os.path.join('models/cv_figs')
    os.makedirs(fig_dir, exist_ok=True)
    make_plots(ds1='wt', ds2='wt_16mic', savefig=True, fig_dir=fig_dir, display=False)
    # plot_empirical_occupancy_2datasets()
    # plot_weight_magnitudes()
    # plot_weights()
    # plot_legends(0, 0, savefig=True, fig_dir=fig_dir, display=False)
