####################################

# Usage: python comparison_cv.py

####################################

import glob
import random
import joblib
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
import pandas as pd

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


# def loadCV_LLs(path, model_prefix, num_states):
#     model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
#     train_lps = []
#     test_lps = []
#     for _ in model_pkl_paths:
#         pkl, _, _ = utils.load_specific_path(_)
#         if pkl is None:
#             continue
#         train_lps.append(pkl['train_data']['train_lp'].item())
#         test_lps.append(pkl['test_data']['test_lp'].item())
#     return np.array(train_lps), np.array(test_lps)


def loadCV_Scores(path, model_path_prefixes, num_states_configs, precomputed=False, score_type=None):
    """
    :param score_type: 'r2' or 'pearson' or 'll'
    """
    from utilities.io import load_specific_path

    if precomputed:
        scores_dict = joblib.load(f'{path}_all_scores_dict.pkl')
        train_scores = scores_dict[model_path_prefix][num_states]['train'][score_type]
        test_scores = scores_dict[model_path_prefix][num_states]['test'][score_type]
        # print('train', score_type, f'num_states={num_states}', train_scores)
        # print('test', score_type, f'num_states={num_states}', test_scores)
        return train_scores, test_scores

    rows = []
    for model_path_prefix in model_path_prefixes:
        model_path, prefix, datefilter = model_path_prefix
        for num_states in num_states_configs:
            model_pkl_paths = sorted(glob.glob(f'models/{model_path}/{prefix}_{num_states}_cv/{datefilter}**/'))
            # random.shuffle(model_pkl_paths)

            c = 0
            for _ in model_pkl_paths:
                pkl, data_config_pkl, model_config = load_specific_path(_)
                if pkl is None: continue
                datasplit_seed = model_config['datasplit_seed']
                init_seed = model_config['seed']
                name = prefix   # model_path_prefix.split('/')[1]
                for group in ['train', 'test']:
                    r2_score = pkl[f'{group}_data'][f'{group}_score'] * 100
                    pearson_score = pkl[f'{group}_data'][f'{group}_pearson']
                    l2_penalty = model_config['l2_penalty']
                    factor_bits_per_sec = data_config_pkl['effective_fps']/np.log(2)
                    ll_score = pkl[f'{group}_data'][f'{group}_lp'].item() * factor_bits_per_sec
                    row = [name, num_states, group, datasplit_seed, init_seed, r2_score, pearson_score, ll_score, l2_penalty, _]
                    rows.append(row)
                c += 1
                if c == 10: break
    columns=['model', 'num_states', 'group', 'datasplit_seed', 'init_seed', 'r2_score', 'pearson_score', 'll_score', 'l2_penalty', 'path']
    scores_df = pd.DataFrame(rows, columns=columns)
    print(scores_df)
    return scores_df


def plot_ll_scores(
    df, 
    model_prefixes=['id-glm-hmm'], 
    # groups=['train', 'test'], 
    num_states=[2, 3, 4, 5, 6, 7, 8, 10], 
    plot_type='mean_sem',   # 'mean_sem' or 'all_points'
    # group_by_seed=None,     # None, 'datasplit_seed', or 'init_seed'
    prefix='',
    score_type='ll_score'
):
    """
    Plots ll_score vs num_states for specified models, groups, and seeds.
    """
    np.random.seed(0)

    # 1. Filter for specific states and groups
    # plot_df = df[df['num_states'].isin(num_states)].copy()
    # plot_df = plot_df[plot_df['group'].isin(groups)]
    
    # # 2. Filter and extract model prefixes (to handle names like 'hmm_1', 'glm_base')
    # def get_prefix(model_name):
    #     for prefix in model_prefixes:
    #         if model_name.startswith(prefix):
    #             return prefix
    #     return None
        
    # plot_df['model_prefix'] = plot_df['model'].apply(get_prefix)
    # plot_df = plot_df.dropna(subset=['model_prefix'])

    # plot_df['model_prefix'] = plot_df['model']
    
    # 3. Determine how we are grouping the data for lines/colors
    group_cols = ['model_prefix', 'group']
    # if group_by_seed:
    #     group_cols.append(group_by_seed)
        
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True, sharey=True)
    colors = {
        'lrhmmci_': 'r',
        'id-glm-hmm': 'b',
    }

    labels = {
        'lrhmmci_': 'w/o GLM Transitions',
        'id-glm-hmm': 'w GLM Transitions',
    }
    
    # 4. Group data and plot
    for e, groups in enumerate([['train'], ['test']]):
        ax = axes[e]

        plot_df = df[df['num_states'].isin(num_states)].copy()
        plot_df = plot_df[plot_df['group'].isin(groups)]
        plot_df['model_prefix'] = plot_df['model']

        for name, group_df in plot_df.groupby(group_cols):
            # 'name' is a tuple representing the current group
            label = " | ".join(map(str, name))
            print(label)

            group_df = group_df.sort_values('num_states')

            if plot_type == 'all_points':
                ax.scatter(group_df['num_states'] + np.random.uniform(-0.1, 0.1, size=len(group_df)),
                            group_df[score_type], color=colors[name[0]], edgecolors='none', s=5, alpha=0.4) #, label=label)

            # elif plot_type == 'mean_sem':
            # Calculate mean and standard error of the mean
            stats_df = group_df.groupby('num_states')[score_type].agg(['mean', 'sem']).reset_index()
            stats_df['sem'] = stats_df['sem'].fillna(0) # Handle single-point edge cases

            ax.errorbar(
                stats_df['num_states'], 
                stats_df['mean'], 
                yerr=stats_df['sem'], 
                marker='o', 
                capsize=0,
                # linewidth=2,
                markersize=2,
                label=labels[name[0]],
                color=colors[name[0]],
            )
            # print(stats_df)

        # 5. Formatting and Aesthetics
        ax.set_xticks(num_states)
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel('Number of States')
        if score_type == 'll_score':
            ax.set_ylabel('Normalized LL (bits/s)')
        elif score_type == 'pearson_score':
            ax.set_ylabel(r'Pearson $r$')
        elif score_type == 'r2_score':
            ax.set_ylabel('Var Explained (%)')
        else:
            raise NotImplementedError

        ax.set_title(f'GLM-HMM ({name[1].title()})')

        handles, hlabels = ax.get_legend_handles_labels()
        cleaned_handles = [ h[0] if type(h).__name__ == 'ErrorbarContainer' else h for h in handles]
        ax.legend(cleaned_handles, hlabels, loc='lower right')

        ax.margins(0.1)
        ax.grid(alpha=0.15)
        # ax.set_ylim([90, 280])

    if savefig:
        filename = f'{prefix}_{model_prefixes}_{groups}_{plot_type}_both_wo1.pdf'
        print(f'Saved at: {filename}')
        plt.savefig(f'models/cv_figs/{filename}', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plot_penalty_cv(prefix, df):

    df = df.groupby(["num_states", "l2_penalty", "group"], as_index=False)["ll_score"].mean()

    l2_values = sorted(df["l2_penalty"].unique())
    colors = plt.cm.tab10(range(len(l2_values)))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for color, l2 in zip(colors, l2_values):
        sub = df[df["l2_penalty"] == l2]
        train = sub[sub["group"] == "train"].sort_values("num_states")
        test  = sub[sub["group"] == "test"].sort_values("num_states")
        ax.plot(train["num_states"], train["ll_score"], color=color, linestyle="-", marker=".", label=f"l2={l2} train")
        ax.plot(test["num_states"],  test["ll_score"],  color=color, linestyle=":", marker=".", label=f"l2={l2} test")

    ax.set_xlabel("num_states")
    ax.set_ylabel("Normalized LL (bits/s)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(title="L2 / split")
    ax.margins(0.1)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    if savefig:
        filename = f'{prefix}_penalty.pdf'
        print(f'Saved at: {filename}')
        plt.savefig(f'models/cv_figs/{filename}', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plot_penalty_cv_perstate(prefix, df, filter_numstates=[]):

    df = df.groupby(["num_states", "l2_penalty", "group"], as_index=False)["ll_score"].mean()

    states = sorted(df["num_states"].unique())
    colors = plt.cm.tab10(range(len(states)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for color, ns in zip(colors, states):
        if ns not in filter_numstates: continue
        sub = df[df["num_states"] == ns]
        train = sub[sub["group"] == "train"].sort_values("l2_penalty")
        test  = sub[sub["group"] == "test"].sort_values("l2_penalty")
        ax.plot(train["l2_penalty"], train["ll_score"], color=color, linestyle="-",
                marker=".", label=f"n={ns} / Train")
        ax.plot(test["l2_penalty"],  test["ll_score"],  color=color, linestyle=":",
                marker=".", label=f"n={ns} / Test")

    ax.set_xlabel("l2_penalty")
    ax.set_ylabel("Normalized LL (bits/sec)")
    ax.set_xscale("log")
    ax.legend(title="states / split", fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.margins(0.1)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    if savefig:
        filename = f'{prefix}_penalty_perstate_{filter_numstates}.pdf'
        print(f'Saved at: {filename}')
        plt.savefig(f'models/cv_figs/{filename}', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def return_grouped_ll_scores(
    prefix,
    df, 
    model_prefixes, 
    score_type='ll_score'
    ):

    df['model'] = df['model'].replace('id-glm-hmm', 'GLM Tr')
    df['model'] = df['model'].replace('lrhmmci_', 'Static Tr')

    df_side_by_side = df.pivot_table(
        index=['num_states', 'group'],      # What you want as your rows
        columns='model',   # The categories you want side-by-side
        values='ll_score',         # The numbers to calculate
        aggfunc=['mean', 'sem'] # The math to apply
    )
    df_side_by_side.columns = [f"{model}_{stat}" for stat, model in df_side_by_side.columns]
    df_side_by_side = df_side_by_side.sort_values(by=['group', 'num_states'], ascending=[0, 1])
    df_side_by_side = df_side_by_side.reset_index()
    df_side_by_side['diff_mean(GLM-Static)'] = df_side_by_side['GLM Tr_mean'] - df_side_by_side['Static Tr_mean']
    print(df_side_by_side.to_string(index=False))
    df_side_by_side.to_csv(f'{prefix}_ll_scores_data.csv', index=False)
    return


def scores_dump(path, model_prefixes, num_states_configs):
    hmm_scores_df = loadCV_Scores(path, model_prefixes, num_states_configs)
    print(hmm_scores_df)
    joblib.dump(hmm_scores_df, f'{path}_hmm_scores_df.pkl')
    return


if __name__ == '__main__':
    import sys

    savefig = True
    display = False

    prefix = 'may17'

    # # num_states_configs = [ 1, 2, 3, 4, 5, 6, 7, 8, 10 ]
    # num_states_configs = [ 2, 3, 4, 5, 6, 7, 8, 10 ]
    # # num_states_configs = [2]
    # scores_dump(prefix, [
    #     ('may17_sweepcvcv_wt_female', 'id-glm-hmm', ''),
    #     # ('apr6_bothcv_wt_female_2', 'lrhmmci_', ''),
    #     # ('apr11_bothcv_wt_female', 'id-glm-hmm', '')
    #     ],
    #     num_states_configs)
    # sys.exit(0)

    # df = joblib.load(f'{prefix}_hmm_scores_df.pkl')
    # df.to_csv(f'{prefix}_ll_scores_data.csv', index=False)
    # print(df)
    df = pd.read_csv(f"{prefix}_ll_scores_data.csv")
    # plot_penalty_cv(prefix, df)
    # plot_penalty_cv_perstate(prefix, df)
    plot_penalty_cv_perstate(prefix, df, filter_numstates=[5])
    sys.exit(0)
    print("df ===============\n")

    # 1. Isolate the training data
    train_df = df[df['group'] == 'train']

    # 2. Find the index of the max train ll_score for each setup
    best_train_idx = train_df.groupby(['model', 'num_states', 'datasplit_seed'])['ll_score'].idxmax()

    # 3. Extract the winning combinations of model, states, datasplit, and the resulting best init_seed
    winning_combinations = train_df.loc[best_train_idx, ['model', 'num_states', 'datasplit_seed', 'init_seed']]

    # 4. Merge this back with the original dataframe to get both 'train' and 'test' rows for those specific seeds
    best_runs_df = pd.merge(
        df, 
        winning_combinations, 
        on=['model', 'num_states', 'datasplit_seed', 'init_seed'], 
        how='inner'
    )
    print(best_runs_df)
    print("best_runs_df ===============\n")
    print(winning_combinations)
    print("winning_combinations ===============\n")

    return_grouped_ll_scores(prefix, best_runs_df,
                             model_prefixes=[
                                 'id-glm-hmm',
                                #  'lrhmmci_'
                                 ])
    sys.exit(0)
    print("===============\n")

    plot_ll_scores(best_runs_df, model_prefixes=['id-glm-hmm', 'lrhmmci_'], plot_type='all_points', prefix=prefix)
    # plot_ll_scores(best_runs_df, model_prefixes=['id-glm-hmm', 'lrhmmci_'], groups=['test'], plot_type='all_points', prefix=prefix)
    sys.exit(0)
    plot_ll_scores(df, model_prefixes=['id-glm-hmm'], groups=['train'], plot_type='all_points', prefix=prefix)
    plot_ll_scores(df, model_prefixes=['id-glm-hmm'], groups=['train'], plot_type='mean_sem', prefix=prefix)
    plot_ll_scores(df, model_prefixes=['id-glm-hmm'], groups=['test'], plot_type='all_points', prefix=prefix)
    plot_ll_scores(df, model_prefixes=['id-glm-hmm'], groups=['test'], plot_type='mean_sem', prefix=prefix)
    plot_ll_scores(df, model_prefixes=['lrhmmci_'], groups=['train'], plot_type='all_points', prefix=prefix)
    plot_ll_scores(df, model_prefixes=['lrhmmci_'], groups=['train'], plot_type='mean_sem', prefix=prefix)
    plot_ll_scores(df, model_prefixes=['lrhmmci_'], groups=['test'], plot_type='all_points', prefix=prefix)
    plot_ll_scores(df, model_prefixes=['lrhmmci_'], groups=['test'], plot_type='mean_sem', prefix=prefix)

    # plotCV_same_model_LL_by_fly(path, 'lrhmmci_', num_states_configs, plot_only_test=True, filesuffix='plot_only_test')

