from utilities.io import *
from preprocess.colors import colors_datasource

import numpy as np
import matplotlib.pyplot as plt
from seaborn import xkcd_rgb
from statsmodels.tsa.stattools import acf, pacf
import matplotlib as mpl

mpl.rcParams['font.size'] = 18  # Panel label


def data(model_ckp):
    all_emissions = [*model_ckp['train_data']['train_emissions'], *model_ckp['test_data']['test_emissions']]
    all_stateseq = [*model_ckp['train_data']['train_stateseq'], *model_ckp['test_data']['test_stateseq']]
    all_aux_data = [*model_ckp['train_data']['train_aux_data'], *model_ckp['test_data']['test_aux_data']]
    all_output_mn_std = [*model_ckp['train_data']['train_output_mn_std'], *model_ckp['test_data']['test_output_mn_std']]
    all_aux_mn_std = [*model_ckp['train_data']['train_aux_mn_std'], *model_ckp['test_data']['test_aux_mn_std']]
    return all_emissions, all_stateseq, all_output_mn_std, all_aux_data, all_aux_mn_std


def plot_autocorr(animal, savefig=True, display=False):
    datasets = ['wt', 'ac_both', 'blind_both']
    # datasets = ['wt', 'wt_16mic']

    if animal == 'female':
        vars = [('forward velocity', 0), ('lateral velocity', 1), ('angular velocity', 2)]
    elif animal == 'male':
        vars = [('forward velocity', 0), ('lateral speed', 1)]

    fig, axes = plt.subplots(len(vars), 2, figsize=(14, len(vars) * 4.5 + 1), sharey=True)

    for dataset in datasets:

        model_ckp, data_config, model_config = load_specific_path(model_paths[dataset])
        if model_ckp is None:
            return
        all_emissions, all_stateseq, all_output_mn_std, all_aux_data, all_aux_mn_std = data(model_ckp)
        effective_fps = data_config['effective_fps']
        max_lags = int((data_config['input_raw_each_dim'] / data_config['orig_fps']) * effective_fps)
        num_flies = len(all_emissions)
        color = xkcd_rgb[colors_datasource[dataset]]

        if animal == 'female':
            series = all_emissions
        elif animal == 'male':
            series = all_aux_data
        else:
            raise NotImplementedError

        axi = 0
        for vel_name, e in vars:

            all_acfs = np.zeros((num_flies, max_lags + 1))
            all_pacfs = np.zeros((num_flies, max_lags + 1))

            for i in range(num_flies):
                all_acfs[i, :] = acf(series[i][:, e], nlags=max_lags)
                all_pacfs[i, :] = pacf(series[i][:, e], nlags=max_lags)

            mean_acf = np.mean(all_acfs, axis=0)
            std_acf = np.std(all_acfs, axis=0)
            mean_pacf = np.mean(all_pacfs, axis=0)
            std_pacf = np.std(all_pacfs, axis=0)

            lags = np.arange(max_lags + 1) / effective_fps

            # Plot Population ACF
            ax0 = axes[axi, 0]
            ax0.plot(lags, mean_acf, color=color, marker='o', markersize=3, label=f'{dataset} (n={num_flies})')
            ax0.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, color=color, alpha=0.1)
            ax0.axhline(0, color='black', linestyle='--')
            if ax0.get_subplotspec().is_first_row():
                ax0.set_title(f"Autocorrelation")
                ax0.legend(loc='upper right')
            ax0.set_xlabel("Lag (secs)")
            ax0.set_ylabel(f"{animal} {vel_name}\n(correlation coeff)")
            ax0.margins(y=0.05)
            ax0.grid(alpha=0.2)

            # Plot Population PACF
            ax1 = axes[axi, 1]
            ax1.plot(lags, mean_pacf, color=color, marker='o', markersize=4, label=f'{dataset} (n={num_flies})')
            ax1.fill_between(lags, mean_pacf - std_pacf, mean_pacf + std_pacf, color=color, alpha=0.1)
            ax1.axhline(0, color='black', linestyle='--')
            if ax1.get_subplotspec().is_first_row():
                ax1.set_title(f"Partial Autocorrelation")
                ax1.legend(loc='upper right')
            ax1.set_xlabel("Lag (s)")
            ax1.set_ylabel("(partial correlation)")
            ax1.margins(y=0.05)
            ax1.grid(alpha=0.2)
            axi += 1

    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join('adhocfigures', f'autocorr_{datasets}_{animal}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return


if __name__  == '__main__':

    model_paths = {
        'wt': '../FINAL_WT/20260531_183347_obsidian/',       # wt female
        'wt_16mic': '../FINAL_WT_FRED/20260626_050049_whelp',    # wt female fred
        'ac_both': 'models/general_ac_both/idarhmm_5_cv/20260814_173744_accounting/',         # ac both female
        'blind_both': 'models/general_blind_both/idarhmm_5_cv/20260814_175044_congregation/',  # blind both female
    }
    plot_autocorr('female', display=False, savefig=True)
    plot_autocorr('male', display=False, savefig=True)
