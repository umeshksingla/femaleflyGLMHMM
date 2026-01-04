####################################
# Use to plot metrics for various classes of models (GLMHMM VS GHMM VS SHUFFLED, WT vs WTFRED, male vs female etc.).

# Usage: python comparison_models.py

####################################

import glob
import random
from collections import OrderedDict

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from utilities import utils

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


# def get_mse(pkl, prefix):
#     y_pred = np.concatenate(pkl[f'{prefix}_data'][f'{prefix}_soft_predictions'], axis=0)
#     y_true = np.concatenate(pkl[f'{prefix}_data'][f'{prefix}_emissions'], axis=0)
#     return mean_squared_error(y_true, y_pred)


def get_mse_by_fly(pkl, prefix):
    mse_by_fly = []
    for s in range(len(pkl[f'{prefix}_data'][f'{prefix}_emissions'])):
        y_pred = pkl[f'{prefix}_data'][f'{prefix}_soft_predictions'][s]
        y_true = pkl[f'{prefix}_data'][f'{prefix}_emissions'][s]
        mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
        mse_by_fly.append(mse)
    return np.array(mse_by_fly)


def load_Scores(model_pkl_paths, score_type):
    """
    :param score_type: 'r2_fly' or 'pearson_fly' or 'll_fly' or 'mse_fly'
    """
    train_scores_dict = OrderedDict()
    test_scores_dict = OrderedDict()
    for p in model_pkl_paths:
        pkl, data_config_pkl, _ = utils.load_specific_path(model_pkl_paths[p])
        if pkl is None:
            continue
        if score_type == 'r2_fly':
            train_score = pkl['train_data']['train_score_by_fly'] * 100
            test_score = pkl['test_data']['test_score_by_fly'] * 100
        elif score_type == 'pearson_fly':
            train_score = pkl['train_data']['train_pearson_by_fly']
            test_score = pkl['test_data']['test_pearson_by_fly']
            print(train_score, test_score)
        elif score_type == 'pearson_o_fly':
            train_score = pkl['train_data']['train_correlation_max_by_o_by_fly_soft'][1]
            test_score = pkl['test_data']['test_correlation_max_by_o_by_fly_soft'][1]
            print(train_score, test_score)
        elif score_type == 'll_fly':
            factor_bits_per_sec = data_config_pkl['effective_fps']/np.log(2)
            train_score = pkl['train_data']['train_lps_by_fly'] * factor_bits_per_sec
            test_score = pkl['test_data']['test_lps_by_fly'] * factor_bits_per_sec
        elif score_type == 'mse_fly':
            train_score = get_mse_by_fly(pkl, 'train')[:, 0]
            test_score = get_mse_by_fly(pkl, 'test')[:, 0]
        else:
            raise Exception(f'Unsupported score type "{score_type}".')
        train_scores_dict[p] = train_score
        test_scores_dict[p] = test_score
    return train_scores_dict, test_scores_dict


def plotCV_diff_model_Score_by_fly(paths, plot_only_test=False, filesuffix='', score_type='r2_fly'):
    """
    """

    if score_type not in ['r2_fly', 'pearson_fly', 'll_fly', 'mse_fly', 'pearson_o_fly']:
        raise Exception(f'Unsupported score type "{score_type}".')

    plt.figure(figsize=(len(paths)*1.5+1.5, 4.5), constrained_layout=True)
    ms = 5

    train_scores_dict, test_scores_dict = load_Scores(paths, score_type=score_type)
    assert len(train_scores_dict) == len(paths)
    assert len(test_scores_dict) == len(paths)

    i = 0
    for m in train_scores_dict:

        print(f"{m}: Train: {len(train_scores_dict[m])} Test:{len(test_scores_dict[m])}")

        if not plot_only_test:
            train_jitter = np.random.uniform(-0.1, 0.1, size=len(train_scores_dict[m]))
            plt.plot(i+train_jitter -0.2, train_scores_dict[m], 'ko', mfc='none', markersize=ms, label='Train' if i == 0 else None)
            plt.errorbar(i, np.mean(train_scores_dict[m]), yerr=np.std(train_scores_dict[m]), color='k', fmt='o', capsize=0)

        test_jitter = np.random.uniform(-0.1, 0.1, size=len(test_scores_dict[m]))
        plt.plot(i+test_jitter + 0.2, test_scores_dict[m], 'ko', markersize=ms, label='Held-out' if i == 0 else None)
        plt.errorbar(i + 0.4, np.mean(test_scores_dict[m]), yerr=np.std(test_scores_dict[m]), color='k', fmt='o', capsize=0)

        i+=1

    if score_type == 'r2_fly':
        plt.ylabel('Var Explained (%)')
    elif score_type == 'pearson_fly':
        plt.ylabel(r'Pearson $r$')
        plt.ylim(-0.05, 0.5)
        plt.legend(loc='lower right')
        plt.axhline(y=0, c='k', ls=':', lw=2)
    elif score_type == 'pearson_o_fly':
        plt.ylabel(r'Pearson[o] $r$')
    elif score_type == 'mse':
        plt.ylabel('MSE')
    elif score_type == 'mse_fly':
        plt.ylabel('MSE')
    elif score_type == 'll_fly':
        plt.ylabel(r'Normalized LL (bits/s)')
        # plt.yscale('symlog')
    else:
        raise Exception(f'Unsupported score type "{score_type}".')

    # plt.xlabel('Number of states')
    plt.xticks(list(range(i)), list(paths.keys()))

    # plt.title(model_prefix.upper())
    # plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'models/diffmodels_{score_type}_cv{filesuffix}_{list(paths.keys())}.pdf', bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return


if __name__ == '__main__':

    savefig = True
    display = True

    # model_paths = OrderedDict({
    #     'GLM': 'models/general_wt/lr_1_cv/20260102_171428_ladder',  # GLM            !!!!!!!!!!!!!! NEEDS REPLACING
    #     '5-state\nGLM-HMM': '../paper figs/FINAL WT/20260101_235805_duration',  # 5-GLM-HMM
    #     # '': 'models/general_wt_shuffled/id-glm-hmm_5_cv/20251229_041412_rice',   # 5-GLM-HMM shuffled             !!!!!!!!!!!!!! NEEDS COMPUTING AS WELL AS REPLACING
    #     '5-state\nHMM': 'models/general_wt/ghmm_5_cv/20260102_143703_fireman',  # 5-GHMM no inputs
    #     # '5-state\nGLM-HMM\n(Second Dataset)': 'models__/final_wt_fred/20251229_040811_costume',  # 5-GLM-HMM fred
    #     # '': 'models/general_wt/ghmm_5_cv/20260102_143703_fireman',   # 5-GHMM fred no inputs
    #     '5-state\nGLM-HMM\n(accel.)': 'models/general_wt_acc/id-glm-hmm_5_cv/20260102_153816_remains',  # 5-GLM-HMM predict acceleration
    #     '5-state\nHMM\n(accel.)': 'models/general_wt_acc/ghmm_5_cv/20260102_153345_coffee',  # 5-GHMM predict acceleration no inputs
    # })
    # plotCV_diff_model_Score_by_fly(model_paths, plot_only_test=False, filesuffix='', score_type='mse_fly')
    #
    # model_paths = {
    #     # '': 'models/general_wt/lr_1_cv/20260102_171428_ladder',  # GLM female
    #     'Female': '../paper figs/FINAL WT/20260101_235805_duration',  # 5-GLM-HMM female
    #     'Male': 'models/general_wt_male/20260102_034454_hypothermia',  # male
    # }
    # plotCV_diff_model_Score_by_fly(model_paths, plot_only_test=False, filesuffix='', score_type='mse_fly')

    # model_paths = {
    #     'Dataset 1': 'models/general_wt/20251229_041412_rice',  # 5-GLM-HMM female
    #     'Dataset 2': 'models__/final_wt_fred/20251229_040811_costume',  # 5-GLM-HMM female fred
    # }
    model_paths = {
        'Dataset 1': '../paper figs/FINAL WT/20260101_235805_duration',  # 5-GLM-HMM female
        'Dataset 2': '../paper figs/FINAL WT FRED/20260102_135949_spandex',  # 5-GLM-HMM female fred
    }

    plotCV_diff_model_Score_by_fly(model_paths, plot_only_test=False, filesuffix='', score_type='pearson_fly')

