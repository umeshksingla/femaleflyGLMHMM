import seaborn as sns

from plotting.plots import COLORS, EC
from utilities.io import *
from utilities.utils import update_labels


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


if __name__ == '__main__':
    leap_model_dir = 'models/general_wt_lr/lr_1_cv/20251025_175656_graduate'
    new16mic_model_dir = 'models/general_wt_fred_lr/lr_1_cv/20251025_175736_tutu'

    fig_dir = os.path.join('models/comparison2datasets')
    make_plots(savefig=True, fig_dir=fig_dir, display=True)
