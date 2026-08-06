from utilities.io import *
from preprocess.colors import *

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf


def plot_autocorr(model_dir, savefig=True, display=False, override_fig_dir=True):

    model_ckp, data_config, model_config = load_specific_path(model_dir)
    if model_ckp is None:
        return

    all_emissions = [*model_ckp['train_data']['train_emissions'], *model_ckp['test_data']['test_emissions']]
    all_aux_data = [*model_ckp['train_data']['train_aux_data'], *model_ckp['test_data']['test_aux_data']]
    print(len(all_emissions), all_emissions[0].shape)
    print(len(all_aux_data), all_aux_data[0].shape)

    effective_fps = data_config['effective_fps']
    max_lags = 90
    num_flies = len(all_emissions)

    if animal == 'female':
        series = all_emissions
        vars = [('forward velocity', 0), ('lateral velocity', 1), ('angular velocity', 2)]
        color = EC
    elif animal == 'male':
        series = all_aux_data
        vars = [('forward velocity', 0), ('lateral speed', 1), ('pulse', 6), ('tap', 8)][:2]
        color = IC

    fig, axes = plt.subplots(len(vars), 2, figsize=(18, len(vars)*5+1), sharey=True)

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
        ax0.plot(lags, mean_acf, color=color, marker='o', markersize=4, label='Mean ACF')
        ax0.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, color=color, alpha=0.2, label='±1 Std Dev')
        ax0.axhline(0, color='black', linestyle='--')
        if ax0.get_subplotspec().is_first_row():
            ax0.set_title(f"Population ACF: Total Autocorrelation (n={num_flies}) flies")
            ax0.legend(loc='upper right')
        ax0.set_xlabel("Lag (secs)")
        ax0.set_ylabel("Correlation Coefficient")
        ax0.margins(y=0.05)
        ax0.annotate(f'{animal} {vel_name}', xy=(0, 0.5), xytext=(-100, 0), textcoords='offset points', ha='right', va='center', rotation=90, color=color)
        ax0.grid(alpha=0.3)

        # Plot Population PACF
        ax1 = axes[axi, 1]
        ax1.plot(lags, mean_pacf, color=color, marker='o', markersize=4, label='Mean PACF')
        ax1.fill_between(lags, mean_pacf - std_pacf, mean_pacf + std_pacf, color=color, alpha=0.2, label='±1 Std Dev')
        ax1.axhline(0, color='black', linestyle='--')
        if ax1.get_subplotspec().is_first_row():
            ax1.set_title(f"Population PACF: Partial Autocorrelation (n={num_flies}) flies")
            ax1.legend(loc='upper right')
        ax1.set_xlabel("Lag (secs)")
        ax1.set_ylabel("Partial Correlation")
        ax1.margins(y=0.05)
        ax1.grid(alpha=0.3)

        axi += 1

    plt.tight_layout()
    fig.savefig(os.path.join('.', f'autocorr_{animal}_{dataset}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    return


if __name__  == '__main__':

    animal = 'male'
    dataset = 'wt_fred'

    if dataset == 'wt':
        model_pkl_path = f'../FINAL_WT/20260531_183347_obsidian/'  # for wt
    elif dataset == 'wt_fred':
        model_pkl_path = f'../FINAL_WT_FRED/20260626_050049_whelp'    # for wt_fred
    plot_autocorr(model_pkl_path)
