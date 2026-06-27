####################################

# Usage: python comparison_cv.py

####################################

import glob
import random
import joblib
from pprint import pprint
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
from scipy import stats
import pandas as pd
import itertools

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
                try:
                    pkl, data_config_pkl, model_config = load_specific_path(_)
                except:
                    print(f'!!! Model does not exist yet. Path: {_}!!!')
                    continue
                if pkl is None: continue
                datasplit_seed = model_config['datasplit_seed']
                init_seed = model_config['seed']
                name = prefix   # model_path_prefix.split('/')[1]
                for group in ['train', 'test']:
                    r2_score = pkl[f'{group}_data'][f'{group}_score'] * 100
                    pearson_score = pkl[f'{group}_data'][f'{group}_pearson']
                    l2_penalty = model_config.get('l2_penalty', None)
                    l1_penalty = model_config.get('l1_penalty', None)
                    split = model_config.get('split', None)
                    factor_bits_per_sec = data_config_pkl['effective_fps']/np.log(2)
                    ll_score = pkl[f'{group}_data'][f'{group}_lp'].item() * factor_bits_per_sec
                    row = [name, num_states, group, datasplit_seed, init_seed, r2_score, pearson_score, ll_score, l1_penalty, l2_penalty, split, _]
                    rows.append(row)
                c += 1
                # if c == 10: break
    columns=['model', 'num_states', 'group', 'datasplit_seed', 'init_seed', 'r2_score', 'pearson_score', 'll_score', 'l1_penalty', 'l2_penalty', 'split', 'path']
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


def plot_penalty_cv_split(prefix, df):
    print("l1_penalty values:", df["l1_penalty"].unique())
    print("l2_penalty values:", df["l2_penalty"].unique())

    # if filter_by == 'l2_penalty':
    #     filtered = df[df["l2_penalty"] == l2_penalty]
    #     assert len(df["l1_penalty"].unique()) == 1
    # elif filter_by == 'l1_penalty':
    #     filtered = df[df["l1_penalty"] == l2_penalty]
    #     assert len(df["l2_penalty"].unique()) == 1
    # else:
    #     raise Exception(f'wrong filter_by={filter_by}')

    # if filtered.empty:
    #     raise ValueError(f"No rows found for {filter_by}={l2_penalty}")

    datasplit_seeds = df["datasplit_seed"].unique()
    l2_penalty_values = df["l2_penalty"].unique()
    l1_penalty_values = df["l1_penalty"].unique()
    print("datasplit_seeds values:", datasplit_seeds)
    print("l1_penalty values:", l1_penalty_values)
    print("l2_penalty values:", l2_penalty_values)

    fig, axes = plt.subplots(
        len(l2_penalty_values) * len(l1_penalty_values),
        len(datasplit_seeds),
        figsize=(6*len(datasplit_seeds), 6*len(l2_penalty_values) * len(l1_penalty_values)),
        sharey=True,
        )

    if len(datasplit_seeds) == 1:
        axes = [axes]

    for il, (l2_penalty, l1_penalty) in enumerate(itertools.product(l2_penalty_values, l1_penalty_values)):
        # for il1, l1_penalty in enumerate(l1_penalty_values):
            # ax = axes[il2, il1]
            # print("onto l2_penalty, l1_penalty", l2_penalty, l1_penalty)
            filtered = df[(df['l2_penalty'] == l2_penalty) & (df['l1_penalty'] == l1_penalty)]
            print(filtered)
            for id, datasplit_seed in enumerate(datasplit_seeds):
                ax = axes[il, id]
                datasplit_df = filtered[filtered["datasplit_seed"] == datasplit_seed]
                for split in [0, 1]:
                    split_df = datasplit_df[datasplit_df["split"] == split]
                    for group, gdf in split_df.groupby("group"):
                        # if group != 'train':
                        #     continue
                        plot_df = gdf.groupby("num_states")["ll_score"].mean().reset_index()
                        plot_df = plot_df.sort_values("num_states")
                        ax.plot(
                            plot_df["num_states"] + np.random.uniform(-0.1, 0.1, len(plot_df["ll_score"])),
                            plot_df["ll_score"],
                            marker={'train': 'o', 'test': '*'}[group],
                            label=f'{group} / split = {split}',
                            color={0: "steelblue", 1: "tomato"}[split],
                            linestyle={'train': '-', 'test': ':'}[group],
                        )
                ax.set_title(f"seed={datasplit_seed} L2={l2_penalty} L1={l1_penalty}")
                ax.set_xlabel("Number of States")
                ax.set_xticks([3, 4, 5, 6, 7], labels=[3, 4, 5, 6, 7])
                ax.set_xlim(3, 7)
                if ax.get_subplotspec().is_first_col():
                    ax.set_ylabel("Normalized LL (bits/sec)")
                if il == 0 and id == 0:
                    ax.legend(title="Group")
                ax.margins(0.1)
                ax.grid(alpha=0.15)

    # plt.suptitle(f"l2_penalty: {l2_penalty} | l1_penalty: {l1_penalty}")
    plt.tight_layout()
    if savefig:
        filename = f'{prefix}_l1l2penalties_datasplitseeds_split.pdf'
        print(f'Saved at: {filename}')
        plt.savefig(f'models/cv_figs/{filename}', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plot_penalty_cv_2models(prefix, df):

    # # ── Split into regularized / unregularized ────────────────────────────────────
    # reg_df   = df[df["l1_penalty"].notna() & df["l2_penalty"].notna()].copy()   # id-glm-hmm_ci
    # unreg_df = df[df["l1_penalty"].isna()  & df["l2_penalty"].isna()].copy()    # lrhmmci_

    reg_df   = df[df["model"] == 'id-glm-hmm'].copy()   # id-glm-hmm_ci
    unreg_df = df[df["model"] == 'lrhmmci_'].copy()    # lrhmmci_

    splits    = sorted(df["datasplit_seed"].unique())
    n_splits  = len(splits)
    groups    = ["train", "test"]

    # ── Best regularized (IDGLMHMM) combo per (split, group) ─────────────────────────────────
    # Step 1: find best (l1, l2) per split using test performance only
    best_test_combo = (
        reg_df[reg_df["group"] == "test"]
        .sort_values("ll_score", ascending=False)
        .groupby("datasplit_seed", as_index=False)
        .first()[["datasplit_seed", "l1_penalty", "l2_penalty"]]
    )
    
    # Step 2: retrieve both train and test LL for that best combo
    best_reg = best_test_combo.merge(reg_df, on=["datasplit_seed", "l1_penalty", "l2_penalty"])

    print(">> best_reg ========")
    print(best_reg)
    pd.set_option('max_colwidth', None)
    print(best_reg[['l1_penalty', 'l2_penalty', 'path']])
    print(">> =================")
    print(">> unreg ========")
    print(unreg_df)
    print(">> =================")

    # # ── Pivot for heatmaps: mean ll_score over splits for each (l1, l2) ───────────
    # l1_vals = sorted(reg_df["l1_penalty"].unique())
    # l2_vals = sorted(reg_df["l2_penalty"].unique())

    def make_heatmap_pivot(split, grp):
        sub = reg_df[(reg_df["datasplit_seed"] == split) & (reg_df["group"] == grp)]
        return sub.pivot_table(index="l2_penalty", columns="l1_penalty",
                            values="ll_score", aggfunc="mean")


    # ── Figure layout ─────────────────────────────────────────────────────────────
    #   Row 0        : bar comparison (best reg vs unreg) — train & test side by side
    #   Rows 1-n     : heatmaps per split (train | test)
    n_heatmap_rows = n_splits
    fig = plt.figure(figsize=(6 * 2, 4 + 4 * n_heatmap_rows))
    gs  = gridspec.GridSpec(
        1 + n_heatmap_rows, 2,
        figure=fig,
        hspace=0.5, wspace=0.35,
        height_ratios=[1.2] + [1] * n_heatmap_rows
    )

    colors = {"train": "#4C72B0", "test": "#DD8452"}
    x      = np.arange(n_splits)
    width  = 0.3

    # Compute global y-limits across both groups for a shared scale
    all_bar_vals = []
    for grp in groups:
        for s in splits:
            all_bar_vals += list(unreg_df[(unreg_df["datasplit_seed"] == s) & (unreg_df["group"] == grp)]["ll_score"].values)
            all_bar_vals += list(best_reg[(best_reg["datasplit_seed"] == s) & (best_reg["group"] == grp)]["ll_score"].values)
    all_bar_vals = [v for v in all_bar_vals if not np.isnan(v)]
    y_min = min(all_bar_vals)
    y_max = max(all_bar_vals)
    y_pad = (y_max - y_min) * 0.1
    shared_ylim = (y_min - y_pad, y_max + y_pad)

    # ── Row 0: bar charts ─────────────────────────────────────────────────────────
    for col, grp in enumerate(groups):
        print(f"Group {grp}")
        ax = fig.add_subplot(gs[0, col])

        unreg_vals = [
            unreg_df[(unreg_df["datasplit_seed"] == s) & (unreg_df["group"] == grp)]["ll_score"].values
            for s in splits
        ]
        assert (len(unreg_vals) == n_splits)
        unreg_means = [v.mean() if len(v) else np.nan for v in unreg_vals]

        best_reg_vals = [
            best_reg[(best_reg["datasplit_seed"] == s) & (best_reg["group"] == grp)]["ll_score"].values
            for s in splits
        ]
        assert (len(best_reg_vals) == n_splits)
        best_reg_means = [v[0] if len(v) else np.nan for v in best_reg_vals]
        print("best_reg_means", best_reg_means)

        ax.bar(x - width / 2, unreg_means,  width, label="w/o GLM Transitions", color=colors[grp], alpha=0.5)
        ax.bar(x + width / 2, best_reg_means, width, label="w GLM Transitions",  color=colors[grp], alpha=0.95)

        ax.set_xticks(x)
        xticklabels = []
        for s_i, s in enumerate(splits):
            combo = best_test_combo[best_test_combo["datasplit_seed"] == s][["l1_penalty", "l2_penalty"]].values
            if len(combo):
                l1, l2 = combo[0]
                print(f"Split {s}:\t\t(L1={l1:.0e}, L2={l2:.0e})")
                # xticklabels.append(f"Split {s}\n(L1={l1:.0e},\nL2={l2:.0e})")
                xticklabels.append(f"s{s}")
            else:
                xticklabels.append(f"Split {s}")
        ax.set_xticklabels(xticklabels, fontsize=9, rotation=90)
        ax.set_ylabel("Normalized LL (bits/s)", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylim(shared_ylim)
        ax.set_title(f"{grp.capitalize()} LL — Static-Tr vs GLM-Tr", fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Rows 1-n: heatmaps ────────────────────────────────────────────────────────
    for row, split in enumerate(splits, start=1):
        for col, grp in enumerate(groups):
            ax  = fig.add_subplot(gs[row, col])
            piv = make_heatmap_pivot(split, grp)

            im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="LL Score")
            cbar.set_label("Normalized LL (bits/s)", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([f"{v:.0e}" for v in piv.columns], fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels([f"{v:.0e}" for v in piv.index], fontsize=7)
            ax.set_xlabel("L1 Penalty", fontsize=8)
            ax.set_ylabel("L2 Penalty", fontsize=8)
            ax.set_title(f"Split {split} — {grp.capitalize()} LL Heatmap", fontsize=9)

            # annotate cells
            for i in range(len(piv.index)):
                for j in range(len(piv.columns)):
                    val = piv.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if val < piv.values.mean() else "black")

    fig.suptitle("Model LL Comparison Across Splits", fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    if savefig:
        filename = f'{prefix}_{dataset}_l1l2penalties_datasplitseeds_2models.pdf'
        print(f'Saved at: {filename}')
        plt.savefig(f'models/cv_figs/{filename}', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plot_penalty_cv_datasplitseeds(prefix, df):

    idglmhmm_df   = df[df["model"] == 'id-glm-hmm'].copy()   # id-glm-hmm_ci
    lrhmm_df = df[df["model"] == 'lrhmmci_'].copy()          # lrhmmci_
    # ghmm_df = df[df["model"] == 'ghmm'].copy()               # ghmm

    splits = sorted(set(idglmhmm_df["datasplit_seed"]) & set(lrhmm_df["datasplit_seed"]))   # get splits that were successful on both models

    # ── Best regularized (IDGLMHMM) combo per (split, group) ─────────────────────────────────
    # Step 1: find best (l1, l2) per split using test performance only
    best_test_combo = (
        idglmhmm_df[idglmhmm_df["group"] == "test"]
        .sort_values("ll_score", ascending=False)
        .groupby("datasplit_seed", as_index=False)
        .first()
    )

    print("best_test_combo")
    best_paths = best_test_combo['path'].tolist()
    print(best_paths)
    print([p.split('/')[-2] for p in best_paths])

    best_test_combo = best_test_combo[["datasplit_seed", "l1_penalty", "l2_penalty"]]
    
    # Step 2: retrieve both train and test LL for that best combo
    best_idglmhmm_df = best_test_combo.merge(idglmhmm_df, on=["datasplit_seed", "l1_penalty", "l2_penalty"])

    grp = 'test'

    lrhmm_vals = np.array([
        lrhmm_df[(lrhmm_df["datasplit_seed"] == s) & (lrhmm_df["group"] == grp)]["ll_score"].values
        for s in splits
    ])[:, 0]

    # ghmm_vals = np.array([
    #     ghmm_df[(ghmm_df["datasplit_seed"] == s) & (ghmm_df["group"] == grp)]["ll_score"].values
    #     for s in splits
    # ])[:, 0]

    best_idglmhmm_vals = [
        best_idglmhmm_df[(best_idglmhmm_df["datasplit_seed"] == s) & (best_idglmhmm_df["group"] == grp)]["ll_score"].values
        for s in splits
    ]
    # best_idglmhmm_vals = [(_ if len(_) else [np.nan]) for _ in best_idglmhmm_vals]
    print("best_idglmhmm_vals", best_idglmhmm_vals)
    best_idglmhmm_vals = np.array(best_idglmhmm_vals)[:, 0]

    print("idglmhmm_vals", best_idglmhmm_vals)
    print("lrhmm_vals", lrhmm_vals)
    # print("ghmm_vals", ghmm_vals)

    diffs = [g - l for g, l in zip(best_idglmhmm_vals, lrhmm_vals)]
    n = len(diffs)
    x = np.arange(n)

    median_d = np.median(diffs)
    iqr_lo   = np.percentile(diffs, 25)
    iqr_hi   = np.percentile(diffs, 75)
    print("median_d", median_d, "iqr", (iqr_lo, iqr_hi))

    CLR_POS = '#2271B2'
    CLR_NEG = '#CC3311'

    fig, ax = plt.subplots(figsize=(5, 5))

    # median line and IQR shading
    ax.axhline(median_d, color='#333333', lw=1.2, ls='-')
    # ax.fill_between([-0.5, n - 0.5], iqr_lo, iqr_hi, color='#333333', alpha=0.12, zorder=1)

    # zero line
    ax.axhline(0, color='black', lw=0.8, ls='--')

    # dots colored by sign
    colors = [CLR_POS if d > 0 else CLR_NEG for d in diffs]
    ax.scatter(x, diffs, color=colors, s=36, zorder=3, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(n)])
    ax.set_xlabel('Data split')
    ax.set_ylabel('LL difference (bits/s)\nGLM Tr − Static Tr')
    ax.margins(0.1)

    if savefig:
        filename = f'{prefix}_{dataset}_datasplitseeds.pdf'
        print(f'Saved at: {filename}')
        plt.savefig(f'models/cv_figs/{filename}', bbox_inches='tight', dpi=300, transparent=True)
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
    joblib.dump(hmm_scores_df, f'{path}_{dataset}_hmm_scores_df.pkl')
    hmm_scores_df.to_csv(f'{path}_{dataset}_hmm_scores_df.csv', index=False)
    return


if __name__ == '__main__':
    import sys

    savefig = True
    display = False

    prefix = 'june26l1l2'
    dataset = 'wt_fred'

    # num_states_configs = [5]
    # scores_dump(prefix, [
    #     # (f'{prefix}_sweepcv_{dataset}_female', 'ghmm', ''),
    #     (f'{prefix}_sweepcv_{dataset}_female', 'lrhmmci_', ''),
    #     (f'{prefix}_sweepcv_{dataset}_female', 'id-glm-hmm', ''),

    #     # ('may23_sweepcv_wt_female', 'id-glm-hmm', ''),
    #     # ('may17_sweepcvcv_wt_female', 'id-glm-hmm', ''),
    #     # ('apr6_bothcv_wt_female_2', 'lrhmmci_', ''),
    #     # ('apr11_bothcv_wt_female', 'id-glm-hmm', '')
    #     ],
    #     num_states_configs)

    df = joblib.load(f'{prefix}_{dataset}_hmm_scores_df.pkl')
    print("df ===============\n")
    print(df)
    # plot_penalty_cv_split(prefix, df)
    plot_penalty_cv_2models(prefix, df)
    plot_penalty_cv_datasplitseeds(prefix, df)
    sys.exit(0)

    df = pd.read_csv(f"{prefix}_hmm_scores_df.csv")
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

