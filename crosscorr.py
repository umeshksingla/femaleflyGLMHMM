from utilities.io import *
from preprocess.colors import colors_datasource

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from seaborn import xkcd_rgb
from statsmodels.graphics.tsaplots import plot_ccf
from statsmodels.tsa.stattools import ccovf, ccf
import matplotlib as mpl

mpl.rcParams['font.size'] = 18  # Panel label


def bi_ccf(x, y, nlags=10):
    # Positive lags: Corr(x[t+k], y[t]) for k >= 0
    ccf_xy = sm.tsa.stattools.ccf(x, y, adjusted=False)[:nlags + 1]

    # Negative lags via swapping arguments: Corr(y[t+k], x[t]) = Corr(x[t], y[t+k]), i.e. x precedes y by k
    ccf_yx = sm.tsa.stattools.ccf(y, x, adjusted=False)[1:nlags + 1]

    # Combine negative lags (reversed order) and positive lags
    full_ccf = np.concatenate([ccf_yx[::-1], ccf_xy])
    full_lags = np.arange(-nlags, nlags + 1)
    return full_lags, full_ccf


def data(model_ckp):
    all_emissions = [*model_ckp['train_data']['train_emissions'], *model_ckp['test_data']['test_emissions']]
    all_stateseq = [*model_ckp['train_data']['train_stateseq'], *model_ckp['test_data']['test_stateseq']]
    all_aux_data = [*model_ckp['train_data']['train_aux_data'], *model_ckp['test_data']['test_aux_data']]
    all_output_mn_std = [*model_ckp['train_data']['train_output_mn_std'], *model_ckp['test_data']['test_output_mn_std']]
    all_aux_mn_std = [*model_ckp['train_data']['train_aux_mn_std'], *model_ckp['test_data']['test_aux_mn_std']]
    return all_emissions, all_stateseq, all_output_mn_std, all_aux_data, all_aux_mn_std


# def plot_crosscorr(model_dir, savefig=True, display=False):
def plot_crosscorr(savefig=True, display=False):

    female_vars = [('female forward velocity', 0), ('|female lateral velocity|', 1), ('|female angular velocity|', 2)]
    male_modalities = {
        'visual': [('male forward velocity', 0), ('male lateral speed', 1)],
        'song': [('pulse', 6), ('sine', 7)],
        'tap': [('tap', 8)],
    }

    ylims = {
        'visual': (-0.3, 0.6),
        'song': (-0.1, 0.15),
        'tap': (-0.1, 0.15),
    }

    # datasets = ['wt', 'wt_16mic']
    datasets = ['wt', 'ac_both', 'blind_both']

    for modal in male_modalities:
        male_vars = male_modalities[modal]
        fig, axes = plt.subplots(len(female_vars) * len(male_vars), 1, figsize=(13, len(female_vars) * len(male_vars) * 4.5 + 1.2), sharey=True)

        for dataset in datasets:

            model_ckp, data_config, model_config = load_specific_path(model_paths[dataset])
            if model_ckp is None:
                continue

            effective_fps = data_config['effective_fps']
            all_emissions, all_stateseq, all_output_mn_std, all_aux_data, all_aux_mn_std = data(model_ckp)
            all_stateseq = [np.zeros(_.shape) for _ in all_stateseq]  # make them all in same state 0
            aux = get_aux_by_state(all_aux_data, all_stateseq, 1, all_aux_mn_std, rescaled=True, keep_btchs_separate=True)
            emissions = get_emissions_by_state(all_emissions, all_stateseq, 1, all_output_mn_std, rescaled=True, effective_fps=effective_fps, keep_btchs_separate=True)
            aux = aux[0]  # all in same state 0
            emissions = emissions[0]
            max_lags = int((data_config['input_raw_each_dim'] / data_config['orig_fps']) * effective_fps)
            print("dataset:", dataset, "max_lags:", max_lags)
            num_flies = len(all_emissions)
            color = xkcd_rgb[colors_datasource[dataset]]

            axi = 0
            for fem_feat_name, fe in female_vars:
                for mal_feat_name, me in male_vars:

                    all_ccfs = np.zeros((num_flies, 2*max_lags+1))

                    for i in range(num_flies):
                        female_series = emissions[i][:, fe]
                        if ('lateral' in fem_feat_name) or ('angular' in fem_feat_name):
                            # One subtlety on order of operations: abs(zscore(AV)) and zscore(abs(AV)) give the same correlation with song,
                            # but only if raw AV has ~zero mean. z(AV) = (AV − μ)/σ, so |z(AV)| = |AV − μ|/σ — a rescaling of distance from the mean, not distance from zero.
                            # If μ ≈ 0 (plausible with no overall L/R bias), that's the same thing as |AV|/σ and the order doesn't matter.
                            # But if raw AV has some systematic bias (mean skewed one way for reasons unrelated to song),
                            # abs-ing the already-z-scored version measures deviation from her average turning, not deviation from not turning.
                            female_series = np.abs(female_series)  # CAUTION!! female_series is zscored. so abs can only be done if raw velocity mean is zero.
                        male_series = aux[i][:, me]
                        print(f"{mal_feat_name}-{i}", np.unique(male_series))
                        if len(np.unique(male_series)) == 1:
                            continue
                        lags, all_ccfs[i, :] = bi_ccf(female_series, male_series, nlags=max_lags)   # Oct 28, 2026. peak at lag > 0: x leads y
                        # So peak at positive lags mean the female comes after the male, and negative lags mean she comes before him

                    mean_acf = np.mean(all_ccfs, axis=0)
                    std_acf = np.std(all_ccfs, axis=0)
                    lags = lags / effective_fps
                    # Plot Population ACF
                    ax0 = axes[axi]
                    ax0.plot(lags, mean_acf, color=color, marker='o', markersize=3, label=f'{dataset} (n={num_flies})')
                    ax0.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, color=color, alpha=0.1)
                    ax0.axhline(0, color='black', linestyle='--')
                    ax0.set_title(f"({fem_feat_name} / {mal_feat_name})")
                    ax0.set_xlabel("Lag (s)")
                    ax0.set_ylabel("Correlation Coefficient")
                    ax0.set_ylim(ylims[modal])
                    ax0.margins(y=0.05)
                    ax0.grid(alpha=0.3)
                    if ax0.get_subplotspec().is_first_row():
                        ax0.legend(loc='upper right')
                    axi += 1

        plt.tight_layout()
        if savefig:
            fig.savefig(os.path.join('adhocfigures', f'crosscorr_{datasets}_{modal}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
        if display:
            plt.show()
        plt.close()
    return


if __name__  == '__main__':

    # for dataset in ['blind_both', 'wt', 'wt_16mic', 'ac_both']:
    #     if dataset == 'wt':
    #         model_pkl_path = '../FINAL_WT/20260531_183347_obsidian/'
    #     elif dataset == 'ac_both':
    #         model_pkl_path = 'models/general_ac_both/idarhmm_5_cv/20260814_173744_accounting/'
    #     elif dataset == 'blind_both':
    #         model_pkl_path = 'models/general_blind_both/idarhmm_5_cv/20260814_175044_congregation/'
    #     elif dataset == 'wt_16mic':
    #         model_pkl_path = '../FINAL_WT_FRED/20260626_050049_whelp'
    #     else:
    #         raise NotImplementedError
    #     plot_crosscorr(model_pkl_path, display=False, savefig=True)

    model_paths = {
        'wt': '../FINAL_WT/20260531_183347_obsidian/',       # wt female
        'wt_16mic': '../FINAL_WT_FRED/20260626_050049_whelp',    # wt female fred
        'ac_both': 'models/general_ac_both/idarhmm_5_cv/20260814_173744_accounting/',         # ac both female
        'blind_both': 'models/general_blind_both/idarhmm_5_cv/20260814_175044_congregation/',  # blind both female
    }
    plot_crosscorr(display=False, savefig=True)
