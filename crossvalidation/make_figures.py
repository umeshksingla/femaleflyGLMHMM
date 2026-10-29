"""
Model-performance figures: Figure 2 (panels b-h) and four supplementary figures.

Usage, from the folder you run the pipeline from (needed only for the one-time wing-flick step below):
    python make_figures.py                  # paths from CONFIG
    python make_figures.py config.json      # or a JSON file with the same keys as CONFIG

Writes to CONFIG['out']:
  - each panel on its own, e.g. fig2b.pdf, figS1a.pdf (+ .png for a quick look)
  - each figure assembled: fig2.pdf (panel a left empty for the schematic), figS1_cv.pdf,
    figS2_per_fly.pdf, figS3_correlations.pdf, figS4_wing_flicks.pdf
PDF text stays editable (TrueType fonts). A panel whose input is missing is skipped with a message.

Inputs (per dataset, set in CONFIG):
  scores        per_fly_scores.csv from collect_folds.py (K = 5)
  by_k          per_fly_by_K.csv from collect_k_sweep.py
  correlations  correlations_by_fly.csv from collect_correlations.py
  results_dir   folder with the tuning JSONs and checkpoints: the penalty curves come from the
                tuning JSONs, the wing-flick results from auxem_model.pkl in each fold's
                id-glm-hmm checkpoint.
The wing-flick F1 by state on training flies (Fig. 2h, left) was not saved by earlier runs of
flick_model.flick_performance. For those folds it is recomputed once from the fold's saved filters
and its model.pkl (slow: one full checkpoint per fold) and cached in flick_train_by_state.json.

Panels, and the Results statements they show:
  Fig. 2b   Held-out log-likelihood per fly: Chance, GLM, GLM-HMM with K = 2-8 and 20 for Dataset 1
  Fig. 2c   Per-fly gain from cue-dependent transitions (full model - fixed transitions), both
  Fig. 2d   Overall Pearson r per held-out fly, both datasets
  Fig. 2e   r by output, training and held-out flies
  Fig. 2f   Wing-flick F1, training and held-out flies
  Fig. 2g   r by state, training and held-out flies
  Fig. 2h   Wing-flick F1 by state, training and held-out flies
  Fig. S1   Cross-validation: (a) validation log-likelihood vs ridge penalty per fold;
            (b) vs transition penalty per fold, with lambda2 = 1e8 as the reference in the Methods;
            (c) held-out gain from cue-dependent transitions by fold (positive in all ten folds);
            (d) the same gain at each K; (e) gain per added state
  Fig. S2   Per fly: (a-b) held-out score under each model (Chance, GLM, HMM without inputs,
            fixed transitions, full) for each dataset;
            (c-d) each fly's gain from cue-dependent transitions, sorted
            (e) Held-out log-likelihood per fly: Chance, GLM, GLM-HMM with K = 2-8, 12, 15 20 for Dataset 2
  Fig. S3   Correlations: (a) held-out r by state for each output (Dataset 1);
            (b-c) Dataset 2 r by output and by state, training and held-out flies
  Fig. S4   Wing flicks: (a) F1 of states only vs states + cues; (b) average precision vs random
            (c) log-likelihood gain over a constant flick rate ;
            (d) per-fly gain from the cues beyond the states;
            (e) precision-recall curves on held-out flies

Summaries next to each column of dots: median and interquartile range for log-likelihoods
(as in the Results), mean and SD for correlations. In the correlation panels each fly keeps one
shade and one sideways position in every column of a panel; shades run from light (best
predicted) to dark, ordered by forward-velocity r in the by-output panels and by overall r in the
others, so a fly's rank in that column can be compared with its rank in the other columns.
"""
import glob
import json
import os
import sys

import joblib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from dynamax.utils.plotting import CMAP, COLORS
from preprocess.colors import *
from collect_folds import BITS, get_k  # noqa: E402

CONFIG = {
    'out': 'figuresclaude',
    'K': 5,                 # states in the main model
    'flick_output': 0,      # index of wing flicking among the auxiliary outputs
    'fps': 30,
    'datasets': {           # rename the files to match yours; leave an entry out to skip its panels
        'Dataset 1': {
            'scores': './models/octladder_cv_wt_female/per_fly_scores_k=5.csv',
            'by_k': './models/octladder_cv_wt_female/per_fly_by_K.csv',
            'correlations': './models/octladder_cv_wt_female/correlations_by_fly_id-glm-hmm_k=5.csv',
            'results_dir': './models/octladder_cv_wt_female',
        },
        'Dataset 2': {
            'scores': './models/octladder_cv_wt_fred_female/per_fly_scores_k=5.csv',
            'by_k': './models/octladder_cv_wt_fred_female/per_fly_by_K.csv',
            'correlations': './models/octladder_cv_wt_fred_female/correlations_by_fly_id-glm-hmm_k=5.csv',
        },
    },
}

# ---------------------------------------------------------------------------------------------- style
STATE_COLORS = COLORS[:5]   # States 1-5: swap in your exact colors
OUTPUT_COLOR = EC     # per-output columns (one shade per fly)
ALL_COLOR = '#8c8c8c'        # 'All' column
FLICK_COLOR = EC
STATES_ONLY_COLOR = '#fbc4b4'
SUMMARY_COLOR = '0.25'
OUTPUT_LABELS = {'forward': 'forward\nvelocity', 'lateral': 'lateral\nvelocity', 'angular': 'angular\nvelocity'}

# plt.rcParams.update({
#     'font.family': 'sans-serif',
#     'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
#     'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'figure.titlesize': 9,
#     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
#     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7,
#     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
#     'lines.linewidth': 1, 'legend.frameon': False,
#     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
# })

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


def new_figure(size):
    try:
        return plt.figure(figsize=size, layout='constrained')
    except TypeError:                       # matplotlib < 3.5
        return plt.figure(figsize=size, constrained_layout=True)


def title(fig, text=''):
    """Every panel gets a title row (blank if no title), which also leaves room for the panel letter."""
    # fig.suptitle(text or ' ', fontsize=plt.rcParams['figure.titlesize'])
    fig.suptitle('')


def mix(color, other, a):
    return tuple((1 - a) * np.array(to_rgb(color)) + a * np.array(to_rgb(other)))


def fly_colors(color, flies, order_by):
    """A shade of `color` per fly: lightest for the fly with the highest `order_by` (a per-fly Series,
    e.g. forward-velocity r) and darkest for the lowest. Passing the same `order_by` for every column
    keeps each fly's shade, so its rank in one column can be compared with the others.
    Flies without a value get the middle shade."""
    rank = order_by.rank(ascending=True, method='first')        # 1 = highest
    n = max(int(rank.count()) - 1, 1)
    cmap = LinearSegmentedColormap.from_list('', [mix(color, 'w', 0.65), color, mix(color, 'k', 0.45)])
    return [cmap(0.1 + 0.9 * (rank[f] - 1) / n) if f in rank.index and np.isfinite(rank[f]) else cmap(0.55)
            for f in flies]


def fly_offsets(flies, width=0.14):
    """A fixed sideways offset per fly, the same in every column."""
    return [np.random.default_rng([int(f), 7]).uniform(-width, width) for f in flies]

 
def strip(ax, x, values, colors='k', summary='median', size=None, offsets=None):
    """Per-fly dots around x, with the summary just to the right: median and interquartile range,
    or mean and SD. `offsets` gives each dot's sideways offset (fly_offsets); else random jitter."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if not ok.any():
        return
    if not isinstance(colors, str):
        colors = [c for c, keep in zip(colors, ok) if keep]
    if offsets is None:
        offsets = np.random.default_rng(int(round(abs(x) * 100)) + 7).uniform(-0.14, 0.14, len(v))
    offsets, v = np.asarray(offsets, float)[ok], v[ok]
    ax.scatter(x - 0.06 + offsets, v, s=size or (12 if len(v) > 30 else 12),
               c=colors, linewidths=0, zorder=2)
    if summary == 'median':
        mid, (lo, hi) = np.median(v), np.percentile(v, [25, 75])
    else:
        mid, lo, hi = v.mean(), v.mean() - v.std(), v.mean() + v.std()
    ax.plot([x + 0.3] * 2, [lo, hi], color=SUMMARY_COLOR, lw=1, zorder=3)
    ax.plot([x + 0.3], [mid], 'o', color=SUMMARY_COLOR, ms=3, zorder=3)


def zero_line(ax):
    ax.axhline(0, color='k', ls=':', lw=0.8, zorder=1)


def ticks(ax, positions, labels, rotation=0):
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation)


def k_positions(ks, start=0.0, gap=0.8):
    """x position for each K, with a gap wherever K jumps by more than one (e.g. 8 -> 20)."""
    pos, x = [], start
    for i, k in enumerate(ks):
        if i and k - ks[i - 1] > 1:
            x += gap
        pos.append(x)
        x += 1
    return pos


def mark_breaks(ax, ks, pos):
    for i in range(1, len(ks)):
        if ks[i] - ks[i - 1] > 1:
            ax.text((pos[i] + pos[i - 1]) / 2, 0, '//', transform=ax.get_xaxis_transform(), ha='center',
                    va='center', clip_on=False, bbox=dict(fc='w', ec='none', pad=0.5)) # fontsize=7


# ---------------------------------------------------------------------------------------------- data
def read_csv(path, **kw):
    if not path:
        return None
    if not os.path.exists(path):
        print(f'  not found: {path} (panels that need it are skipped)')
        return None
    return pd.read_csv(path, **kw)


def read_scores(path):
    """per_fly_scores.csv -> one row per held-out fly, bits/s over Chance for each model."""
    df = read_csv(path, index_col=0)
    if df is None:
        return None
    out = pd.DataFrame({'GLM': df['glm (bits/s)'], 'HMM': df['dghmm (bits/s)'],
                        'fixed': df['lrhmm (bits/s)'], 'full': df['idglmhmm (bits/s)']})
    out['fold'] = df['fold'].values
    return out


def corr_table(corr, measure, split, column=None, output=None):
    """Per-fly r: a Series (column=None) or a fly x state / fly x output table."""
    d = corr[(corr['measure'] == measure) & (corr['split'] == split)]
    if output is not None:
        d = d[d['output'] == output]
    if column is None:
        return d.groupby('fly')['r'].mean()
    t = d.pivot_table(index='fly', columns=column, values='r')
    if column == 'state':
        t.columns = [int(c) for c in t.columns]
    else:
        t = t[[o for o in OUTPUT_LABELS if o in t.columns] + [o for o in t.columns if o not in OUTPUT_LABELS]]
    return t


def tuning_runs(results_dir, k):
    """Latest tuning JSON of each integer fold for this K (skips the fit to all flies)."""
    runs = {}
    for p in sorted(glob.glob(os.path.join(results_dir, '**', '*_tuning.json'), recursive=True),
                    key=os.path.getmtime):
        with open(p) as f:
            t = json.load(f)
        if isinstance(t.get('fold'), int) and get_k(t) == k:
            runs[t['fold']] = t
    return dict(sorted(runs.items()))


def log_lik(y, p):
    return float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def pooled_f1(ms):
    ms = list(ms)
    return f1(*(sum(m[k] for m in ms) for k in ('tp', 'fp', 'fn'))) if ms else np.nan


def train_by_state(ck_dir, r, o, K):
    """Wing-flick counts by state on a fold's training flies, at that fold's threshold."""
    if 'train_by_state' in r:                                   # saved by newer flick_performance
        return {int(k): v for k, v in r['train_by_state'].items()}
    cache = os.path.join(ck_dir, 'flick_train_by_state.json')
    if os.path.exists(cache):
        with open(cache) as f:
            return {int(k): v for k, v in json.load(f).items()}
    print(f'  wing flicks: computing F1 by state on training flies from {ck_dir}/model.pkl (once) ...')
    tr = joblib.load(os.path.join(ck_dir, 'model.pkl'))['train_data']
    missing = [k for k in ('train_inputs', 'train_aux_emissions', 'train_stateseq') if k not in tr]
    if missing:
        raise KeyError(f'{ck_dir}/model.pkl has no {missing}; its train_data keys are {sorted(tr)}')
    X = np.concatenate([np.asarray(a, float) for a in tr['train_inputs']])
    y = np.concatenate([np.asarray(a).reshape(len(a), -1) for a in tr['train_aux_emissions']])[:, o].astype(int)
    z = np.concatenate([np.asarray(a).ravel() for a in tr['train_stateseq']]).astype(int)
    w, b = np.asarray(r['w'], float), np.asarray(r['b'], float)
    logit = np.full(len(X), -np.inf)          # a state without flicks in training has no filter (b = NaN): p = 0
    for k in range(K):
        m = z == k
        if m.any() and np.isfinite(b[k]):
            logit[m] = X[m] @ w[k] + b[k]
    p = np.clip(1 / (1 + np.exp(-logit)), 1e-9, 1 - 1e-9)
    pred = p >= r['threshold']
    counts = lambda m: {'tp': int(np.sum(pred[m] & (y[m] == 1))), 'fp': int(np.sum(pred[m] & (y[m] == 0))),
                        'fn': int(np.sum(~pred[m] & (y[m] == 1))), 'frames': int(m.sum())}
    out = {k: counts(z == k) for k in range(K) if np.any(z == k)}
    for v in out.values():
        v['f1'] = f1(v['tp'], v['fp'], v['fn'])
    check = f1(*(sum(v[c] for v in out.values()) for c in ('tp', 'fp', 'fn')))
    if abs(check - r['train']['f1']) > 0.01:
        print(f'  warning: recomputed training F1 {check:.3f} differs from the saved {r["train"]["f1"]:.3f}; '
              'check that model.pkl holds the same training data and states')
    with open(cache, 'w') as f:
        json.dump(out, f)
    return out


def flick_summary(results_dir, K, o, fps):
    """Wing-flick results pooled over folds, as in flick_model.pool_folds, plus F1 by state."""
    folds = {}
    for fold, t in tuning_runs(results_dir, K).items():
        d = t['checkpoints'].get('id-glm-hmm')
        if d and os.path.exists(os.path.join(d, 'auxem_model.pkl')):
            folds[fold] = (d, joblib.load(os.path.join(d, 'auxem_model.pkl'))[o])
    if not folds:
        print(f'  no auxem_model.pkl found for K={K} folds under {results_dir}')
        return None
    rs = [r for _, r in folds.values()]
    y = np.concatenate([r['test_true'] for r in rs])
    s = {'folds': len(rs), 'y': y, 'rate': float(y.mean()),
         'p': {'full': np.concatenate([r['test_probs'] for r in rs]),
               'states': np.concatenate([r['states_only_test_probs'] for r in rs])}}
    bits = lambda ms: sum(m['ll_gain_nats'] for m in ms) / sum(m['frames'] for m in ms) / np.log(2) * fps
    for name, get in (('full', lambda r: r), ('states', lambda r: r['states_only'])):
        s[name] = {'train_f1': float(np.mean([get(r)['train']['f1'] for r in rs])),
                   'test_f1': pooled_f1(get(r)['test'] for r in rs),
                   'test_ap': float(average_precision_score(y, s['p'][name])),
                   'test_bits': bits([get(r)['test'] for r in rs])}
    s['test_by_state'] = [pooled_f1(r['test_by_state'][k] for r in rs if k in r['test_by_state'])
                          for k in range(K)]
    tbs = [train_by_state(d, r, o, K) for d, r in folds.values()]
    s['train_by_state'] = [float(np.mean([t[k]['f1'] for t in tbs if k in t])) if any(k in t for t in tbs)
                           else np.nan for k in range(K)]
    gains = []                                  # per held-out fly: states + cues vs states only, bits/s
    for r in rs:
        for f in np.unique(r['test_fly']):
            m = r['test_fly'] == f
            yf = r['test_true'][m]
            gains.append((log_lik(yf, r['test_probs'][m]) - log_lik(yf, r['states_only_test_probs'][m]))
                         / m.sum() / np.log(2) * fps)
    s['cue_gain'] = np.array(gains)
    return s


def load(config):
    data = {}
    for name, c in config['datasets'].items():
        print(name)
        d = {'scores': read_scores(c.get('scores')), 'by_k': read_csv(c.get('by_k')),
             'corr': read_csv(c.get('correlations')), 'runs': None, 'flicks': None}
        rd = c.get('results_dir')
        if rd and os.path.isdir(rd):
            d['runs'] = tuning_runs(rd, config['K']) or None
            d['flicks'] = flick_summary(rd, config['K'], config['flick_output'], config['fps'])
        elif rd:
            print(f'  not found: {rd}')
        data[name] = d
    return data


# ---------------------------------------------------------------------------------------------- panels
def panel_k(fig, by_k, k_main):
    """Fig. 2b: held-out log-likelihood per fly for Chance, GLM and the GLM-HMM at each K."""
    title(fig, 'held-out data')
    ax = fig.subplots()
    full = by_k[by_k['model'] == 'idglmhmm'].pivot_table(index='fly', columns='K', values='bits/s')
    glm_rows = by_k[by_k['model'] == 'glm']
    k_glm = k_main if k_main in set(glm_rows['K']) else glm_rows['K'].min()
    glm = glm_rows[glm_rows['K'] == k_glm].set_index('fly')['bits/s']
    ks = sorted(full.columns)
    pos = k_positions(ks, start=2)
    strip(ax, 0, np.zeros(len(glm)))
    strip(ax, 1, glm.values)
    for k, x in zip(ks, pos):
        strip(ax, x, full[k].values)
    ax.grid(True, color='0.9', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ticks(ax, [0, 1] + pos, ['Chance', 'GLM'] + [str(k) for k in ks])
    for lab in ax.get_xticklabels()[:2]:
        lab.set_rotation(90)
    for lab, k in zip(ax.get_xticklabels()[2:], ks):
        lab.set_fontweight('bold' if k == k_main else 'normal')
    mark_breaks(ax, ks, pos)
    ax.annotate('', xy=(pos[-1] + 0.4, -0.13), xytext=(pos[0] - 0.4, -0.13), xycoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='-', lw=1), annotation_clip=False)
    ax.text((pos[0] + pos[-1]) / 2, -0.16, 'GLM-HMM\nnumber of states', transform=ax.get_xaxis_transform(),
            ha='center', va='top')
    ax.set_ylabel('Normalized LL (bits/s)')


def panel_transitions(fig, data):
    """Fig. 2c: per-fly gain from cue-dependent transitions, each dataset."""
    title(fig, 'cue-dependent vs.\nfixed transitions')
    ax = fig.subplots()
    names = [n for n, d in data.items() if d['scores'] is not None]
    for i, n in enumerate(names):
        g = (data[n]['scores']['full'] - data[n]['scores']['fixed']).values
        strip(ax, i, g)
        ax.text(i + 0.1, 1.0, f'{(g > 0).sum()}/{len(g)}', transform=ax.get_xaxis_transform(), ha='center',
                va='bottom')    # fontsize=7
    zero_line(ax)
    ticks(ax, range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    # ax.set_ylabel('Δ normalized LL (bits/s)')
    ax.set_ylabel('Δ normalized LL (bits/s) \n GLM Tr - Fixed Tr')


def panel_r_overall(fig, data):
    """Fig. 2d: overall r per held-out fly, each dataset."""
    title(fig, 'held-out data')
    ax = fig.subplots()
    names = [n for n, d in data.items() if d['corr'] is not None]
    for i, n in enumerate(names):
        r = corr_table(data[n]['corr'], 'overall', 'test')
        strip(ax, i, r.values, colors=fly_colors(ALL_COLOR, r.index, r), offsets=fly_offsets(r.index),
              summary='mean')
    zero_line(ax)
    ticks(ax, range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel('Pearson r')


def train_test_axes(fig, sharey=True):
    axes = fig.subplots(1, 2, sharey=sharey)
    for ax, sub in zip(axes, ('train data', 'held-out data')):
        ax.set_title(sub)
    return dict(zip(('train', 'test'), axes))


def panel_r_by_output(fig, corr, name=None):
    """Fig. 2e: r for each output, training and held-out flies. Flies are shaded by their
    forward-velocity r in that panel (lightest = best predicted), the same shade in every column."""
    title(fig, 'prediction performance' + (f' ({name})' if name else ''))
    flies = corr['fly'].unique()
    for split, ax in train_test_axes(fig).items():
        t = corr_table(corr, 'by_output', split, 'output')
        colors = fly_colors(OUTPUT_COLOR, t.index, t['forward'] if 'forward' in t else t.iloc[:, 0])
        offsets = fly_offsets(t.index)
        for j, out in enumerate(t.columns):
            strip(ax, j, t[out].values, colors=colors, offsets=offsets, summary='mean')
        zero_line(ax)
        ticks(ax, range(t.shape[1]), [OUTPUT_LABELS.get(o, o) for o in t.columns])
    fig.axes[0].set_ylabel('Pearson r')


def panel_r_by_state(fig, corr, K, name=None):
    """Fig. 2g: overall r and r within each state, training and held-out flies. Flies are shaded by
    their overall r in that panel (lightest = best predicted), the same lightness in every column."""
    title(fig, 'prediction performance by state' + (f' ({name})' if name else ''))
    # flies = corr['fly'].unique()
    for split, ax in train_test_axes(fig).items():
        overall = corr_table(corr, 'overall', split)
        strip(ax, 0, overall.values, colors=fly_colors(ALL_COLOR, overall.index, overall),
              offsets=fly_offsets(overall.index), summary='mean')
        t = corr_table(corr, 'by_state', split, 'state')
        for k in range(1, K + 1):
            if k in t:
                strip(ax, k, t[k].values, colors=fly_colors(STATE_COLORS[(k - 1) % len(STATE_COLORS)], t.index, overall),
                      offsets=fly_offsets(t.index), summary='mean')
        zero_line(ax)
        ticks(ax, range(K + 1), ['All'] + [str(k) for k in range(1, K + 1)])
        ax.set_xlabel('State')
    fig.axes[0].set_ylabel('Pearson r')


def panel_r_state_by_output(fig, corr, K):
    """Fig. S3a: held-out r within each state, separately for each output. Flies are shaded by
    their overall held-out r, as in Fig. 2g."""
    title(fig, 'prediction performance by state and output (held-out data)')
    t_all = corr[(corr['measure'] == 'by_state_and_output') & (corr['split'] == 'test')]
    outputs = [o for o in OUTPUT_LABELS if o in set(t_all['output'])]
    axes = np.atleast_1d(fig.subplots(1, len(outputs), sharey=True))
    # flies = corr['fly'].unique()
    overall = corr_table(corr, 'overall', 'test')
    for ax, out in zip(axes, outputs):
        t = corr_table(corr, 'by_state_and_output', 'test', 'state', output=out)
        for k in range(1, K + 1):
            if k in t:
                strip(ax, k, t[k].values, colors=fly_colors(STATE_COLORS[(k - 1) % len(STATE_COLORS)], t.index, overall),
                      offsets=fly_offsets(t.index), summary='mean')
        zero_line(ax)
        ticks(ax, range(1, K + 1), [str(k) for k in range(1, K + 1)])
        ax.set_xlabel('State')
        ax.set_title(OUTPUT_LABELS[out].replace('\n', ' '))
    axes[0].set_ylabel('Pearson r')


def f1_ylim(fs):
    top = max([fs['full']['train_f1'], fs['full']['test_f1']] + list(np.nan_to_num(fs['train_by_state']))
              + list(np.nan_to_num(fs['test_by_state'])))
    return 0, max(0.5, np.ceil(top * 1.25 * 10) / 10)


def panel_flick_f1(fig, fs):
    """Fig. 2f: wing-flick F1 on training and held-out flies."""
    title(fig, 'wing flicking performance')
    for split, ax in train_test_axes(fig).items():
        ax.bar([0], [fs['full'][f'{split}_f1']], width=0.6, color=FLICK_COLOR)
        ticks(ax, [0], ['wing\nflick'])
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(*f1_ylim(fs))
    fig.axes[0].set_ylabel('F1 score')


def panel_flick_by_state(fig, fs, K):
    """Fig. 2h: wing-flick F1 by state on training and held-out flies."""
    title(fig, 'wing flicking performance')
    for split, ax in train_test_axes(fig).items():
        ax.bar(range(1, K + 1), fs[f'{split}_by_state'], width=0.7,
               color=[STATE_COLORS[k % len(STATE_COLORS)] for k in range(K)])
        ticks(ax, range(1, K + 1), [str(k) for k in range(1, K + 1)])
        ax.set_xlabel('State')
        ax.set_ylim(*f1_ylim(fs))
    fig.axes[0].set_ylabel('F1 score')


def penalty_curves(ax, runs, get):
    for fold, t in runs.items():
        x, y, chosen = get(t)
        order = np.argsort(x)
        ax.plot(np.asarray(x)[order], np.asarray(y)[order], color='0.65', lw=0.8, marker='o', ms=2, zorder=2)
        ax.plot([chosen[0]], [chosen[1]], 'o', color='k', ms=3.5, zorder=3)
    ax.set_xscale('log')
    zero_line(ax)


def panel_ridge(fig, runs):
    """Fig. S1a: validation log-likelihood vs ridge penalty, one line per fold; chosen value in black."""
    title(fig, 'emission penalty')
    ax = fig.subplots()

    def get(t):
        rv = t['ridge_val']
        ref = t.get('dghmm_val', rv[max(rv, key=float)])     # vs the HMM (else vs the largest ridge)
        x = [float(k) for k in rv]
        y = [(v - ref) * BITS for v in rv.values()]
        c = t['chosen']['emission_l2']
        return x, y, (c, (rv[f'{c:g}'] - ref) * BITS)
    penalty_curves(ax, runs, get)
    ax.set_xlabel('ridge penalty')
    ax.set_ylabel('validation LL vs. HMM\nwithout inputs (bits/s)')


def panel_l2(fig, runs):
    """Fig. S1b: validation log-likelihood vs transition penalty, relative to fixed transitions."""
    title(fig, 'transition penalty')
    ax = fig.subplots()

    def get(t):
        ch = t['chosen']
        ref = t['ridge_val'][f"{ch['emission_l2']:g}"]
        pts = {tuple(float(a) for a in k.split(',')): v for k, v in t['l1l2_val'].items()}
        pts = {l2: v for (l1, l2), v in pts.items() if l1 == float(ch['l1'])}
        return list(pts), [(v - ref) * BITS for v in pts.values()], (ch['l2'], (pts[float(ch['l2'])] - ref) * BITS)
    penalty_curves(ax, runs, get)
    ax.set_xlabel(r'transition penalty $\lambda_2$')
    ax.set_ylabel('validation LL vs. fixed\ntransitions (bits/s)')


def panel_gain_by_fold(fig, scores):
    """Fig. S1c: held-out gain from cue-dependent transitions, by fold."""
    title(fig, 'cue-dependent vs. fixed transitions')
    ax = fig.subplots()
    g = scores['full'] - scores['fixed']
    folds = sorted(scores['fold'].unique())
    for i, f in enumerate(folds):
        strip(ax, i, g[scores['fold'] == f].values, size=8)
    zero_line(ax)
    ticks(ax, range(len(folds)), [str(f) for f in folds])
    ax.set_xlabel('cross-validation fold')
    # ax.set_ylabel('Δ normalized LL (bits/s)')
    ax.set_ylabel('Δ normalized LL (bits/s) \n GLM Tr - Fixed Tr')


def panel_gain_by_k(fig, by_k):
    """Fig. S1d: held-out gain from cue-dependent transitions at each K."""
    title(fig, 'cue-dependent vs. fixed transitions')
    ax = fig.subplots()
    piv = lambda m: by_k[by_k['model'] == m].pivot_table(index='fly', columns='K', values='bits/s')
    g = piv('idglmhmm') - piv('lrhmm')
    ks = sorted(g.columns)
    pos = k_positions(ks)
    for k, x in zip(ks, pos):
        strip(ax, x, g[k].values)
    zero_line(ax)
    ticks(ax, pos, [str(k) for k in ks])
    mark_breaks(ax, ks, pos)
    ax.set_xlabel('number of states')
    ax.set_ylabel('Δ normalized LL (bits/s) \n GLM Tr - Fixed Tr')


def panel_gain_per_state(fig, by_k):
    """Fig. S1e: held-out gain per added state, between consecutive K (8 -> 20 divided by 12)."""
    title(fig, 'gain per added state')
    ax = fig.subplots()
    full = by_k[by_k['model'] == 'idglmhmm'].pivot_table(index='fly', columns='K', values='bits/s')
    ks = sorted(full.columns)
    for i, (a, b) in enumerate(zip(ks[:-1], ks[1:])):
        strip(ax, i, ((full[b] - full[a]) / (b - a)).values)
    zero_line(ax)
    ticks(ax, range(len(ks) - 1), [f'{a}–{b}' for a, b in zip(ks[:-1], ks[1:])])
    ax.set_xlabel('number of states')
    ax.set_ylabel('gain per added state (bits/s)')


def panel_models_per_fly(fig, scores, name):
    """Fig. S2a-b: each fly's held-out score under each model, lines joining the same fly."""
    title(fig, name)
    ax = fig.subplots()
    cols = ['GLM', 'HMM', 'fixed', 'full']
    M = np.column_stack([np.zeros(len(scores))] + [scores[c].values for c in cols])
    for row in M:
        ax.plot(range(M.shape[1]), row, color='0.7', lw=0.5, alpha=0.7, zorder=1)
    ax.plot(range(M.shape[1]), np.median(M, axis=0), '-o', color='k', ms=3, lw=1.2, zorder=2)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(['Chance', 'GLM', 'HMM, no inputs', 'GLM-HMM, fixed transitions', 'GLM-HMM, full'],
                       rotation=30, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.4, M.shape[1] - 0.6)
    ax.set_ylabel('Normalized LL (bits/s)')


def panel_waterfall(fig, scores, name):
    """Fig. S2c-d: each fly's gain from cue-dependent transitions, sorted."""
    g = np.sort((scores['full'] - scores['fixed']).values)[::-1]
    title(fig, f'{name}: {(g > 0).sum()} of {len(g)} flies improved')
    ax = fig.subplots()
    ax.bar(range(len(g)), g, width=0.8, color=np.where(g > 0, '0.3', '0.75'))
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks([])
    ax.set_xlim(-1, len(g))
    ax.set_xlabel('flies, sorted')
    ax.set_ylabel('Δ normalized LL (bits/s) \n GLM Tr - Fixed Tr')


def panel_flick_vs_states(fig, fs):
    """Fig. S4a: wing-flick F1, states only vs states + cues, training and held-out flies."""
    title(fig, 'F1 score')
    ax = fig.subplots()
    for i, split in enumerate(('train', 'test')):
        ax.bar(i - 0.18, fs['states'][f'{split}_f1'], width=0.34, color=STATES_ONLY_COLOR,
               label='states only' if i == 0 else None)
        ax.bar(i + 0.18, fs['full'][f'{split}_f1'], width=0.34, color=FLICK_COLOR,
               label='states + cues' if i == 0 else None)
    ticks(ax, [0, 1], ['train data', 'held-out data'])
    ax.set_ylim(*f1_ylim(fs))
    ax.set_ylabel('F1 score')
    ax.legend(loc='upper right')


def panel_flick_ap(fig, fs):
    """Fig. S4b: average precision on held-out flies; random prediction = the flick rate."""
    title(fig, 'average precision\n(held-out data)')
    ax = fig.subplots()
    ax.bar([0, 1, 2], [fs['rate'], fs['states']['test_ap'], fs['full']['test_ap']], width=0.6,
           color=['0.8', STATES_ONLY_COLOR, FLICK_COLOR])
    ticks(ax, [0, 1, 2], ['random', 'states\nonly', 'states\n+ cues'])
    ax.set_ylabel('average precision')


def panel_flick_bits(fig, fs):
    """Fig. S4c: log-likelihood gain over a constant flick rate on held-out flies."""
    title(fig, 'gain over a constant\nflick rate (held-out data)')
    ax = fig.subplots()
    ax.bar([0, 1], [fs['states']['test_bits'], fs['full']['test_bits']], width=0.6,
           color=[STATES_ONLY_COLOR, FLICK_COLOR])
    ticks(ax, [0, 1], ['states\nonly', 'states\n+ cues'])
    ax.set_ylabel('LL gain (bits/s)')


def panel_flick_cue_gain(fig, fs):
    """Fig. S4d: per held-out fly, what the cues add beyond each state's flick rate."""
    g = fs['cue_gain']
    title(fig, f'cues beyond states\n({(g > 0).sum()} of {len(g)} flies better)')
    ax = fig.subplots()
    strip(ax, 0, g)
    zero_line(ax)
    ticks(ax, [0], ['held-out flies'])
    ax.set_xlim(-0.6, 0.7)
    ax.set_ylabel('Δ LL, states + cues vs.\nstates only (bits/s)')


def panel_flick_pr(fig, fs):
    """Fig. S4e: precision-recall curves on held-out flies (all folds pooled)."""
    title(fig, 'held-out data')
    ax = fig.subplots()
    top = 0
    for name, color, label in (('states', STATES_ONLY_COLOR, 'states only'), ('full', FLICK_COLOR, 'states + cues')):
        prec, rec, _ = precision_recall_curve(fs['y'], fs['p'][name])
        ax.plot(rec, prec, color=color, lw=1.2, drawstyle='steps-post', label=f'{label} (AP {fs[name]["test_ap"]:.3f})')
        top = max(top, prec[rec >= 0.02].max())      # ignore the few top-ranked frames at near-zero recall
    ax.axhline(fs['rate'], color='0.5', ls='--', lw=0.8, label=f'random ({fs["rate"]:.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1, 1.6 * top))
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.legend(loc='upper right')


# ---------------------------------------------------------------------------------------------- output
def save(fig, out, name):
    for ext in ['pdf']:
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=300 if ext == 'pdf' else 200, bbox_inches='tight', transparent=True)
    plt.close(fig)


def build(config):
    out, K = config['out'], config['K']
    os.makedirs(out, exist_ok=True)
    data = load(config)
    names = list(data)
    d1 = data[names[0]]
    d2 = data[names[1]] if len(names) > 1 else None
    has = lambda d, key: d is not None and d[key] is not None

    # (name, size, draw function or None, inputs that must be present)
    panels = {
        'fig2b': ((8.5, 6), lambda f: panel_k(f, d1['by_k'], K), has(d1, 'by_k')),
        'fig2c': ((4.5, 4), lambda f: panel_transitions(f, data), has(d1, 'scores')),
        'fig2d': ((4, 4), lambda f: panel_r_overall(f, data), has(d1, 'corr')),
        'fig2e': ((9, 4), lambda f: panel_r_by_output(f, d1['corr']), has(d1, 'corr')),
        'fig2f': ((5, 4), lambda f: panel_flick_f1(f, d1['flicks']), has(d1, 'flicks')),
        'fig2g': ((9, 4), lambda f: panel_r_by_state(f, d1['corr'], K), has(d1, 'corr')),
        'fig2h': ((5, 4), lambda f: panel_flick_by_state(f, d1['flicks'], K), has(d1, 'flicks')),
        'figS1a': ((5, 5), lambda f: panel_ridge(f, d1['runs']), has(d1, 'runs')),
        'figS1b': ((5, 5), lambda f: panel_l2(f, d1['runs']), has(d1, 'runs')),
        'figS1c': ((6, 5), lambda f: panel_gain_by_fold(f, d1['scores']), has(d1, 'scores')),
        'figS1d': ((7, 5), lambda f: panel_gain_by_k(f, d1['by_k']), has(d1, 'by_k')),
        'figS1e': ((7, 5), lambda f: panel_gain_per_state(f, d1['by_k']), has(d1, 'by_k')),
        'figS2a': ((8, 7), lambda f: panel_models_per_fly(f, d1['scores'], names[0]), has(d1, 'scores')),
        'figS2b': ((6, 7), lambda f: panel_models_per_fly(f, d2['scores'], names[1]), has(d2, 'scores')),
        'figS2c': ((8, 5), lambda f: panel_waterfall(f, d1['scores'], names[0]), has(d1, 'scores')),
        'figS2d': ((6, 5), lambda f: panel_waterfall(f, d2['scores'], names[1]), has(d2, 'scores')),
        'figS2e': ((8.5, 6), lambda f: panel_k(f, d2['by_k'], K), has(d2, 'by_k')),
        'figS3a': ((7.0, 2.4), lambda f: panel_r_state_by_output(f, d1['corr'], K), has(d1, 'corr')),
        'figS3b': ((9, 4), lambda f: panel_r_by_output(f, d2['corr'], names[1]), has(d2, 'corr')),
        'figS3c': ((9, 4), lambda f: panel_r_by_state(f, d2['corr'], K, names[1]), has(d2, 'corr')),
        'figS4a': ((5, 4), lambda f: panel_flick_vs_states(f, d1['flicks']), has(d1, 'flicks')),
        'figS4b': ((5, 5), lambda f: panel_flick_ap(f, d1['flicks']), has(d1, 'flicks')),
        'figS4c': ((5, 5), lambda f: panel_flick_bits(f, d1['flicks']), has(d1, 'flicks')),
        'figS4d': ((5, 5), lambda f: panel_flick_cue_gain(f, d1['flicks']), has(d1, 'flicks')),
        'figS4e': ((5, 5), lambda f: panel_flick_pr(f, d1['flicks']), has(d1, 'flicks')),
    }
    failed = []

    def draw(name, fig):
        try:
            panels[name][1](fig)
            return True
        except Exception as e:                       # keep going; report at the end
            failed.append(f'{name}: {e!r}')
            return False

    print('\nPanels:')
    for name, (size, _, ok) in panels.items():
        if not ok:
            print(f'  {name}: skipped (input missing)')
            continue
        fig = new_figure(size)
        if draw(name, fig):
            save(fig, out, name)
            print(f'  {name}')
        else:
            plt.close(fig)

    if failed:
        print('\nFailed panels (the rest were made):')
        for f in failed:
            print('  ' + f)
    return data


def report(data, K):
    """Numbers to compare with the Results text."""
    print('\nNumbers behind the panels:')
    for name, d in data.items():
        s = d['scores']
        if s is not None:
            g = s['full'] - s['fixed']
            print(f'  {name}: cue-dependent transitions, median {g.median():.3f} bits/s, '
                  f'{(g > 0).sum()} of {len(g)} flies better; full model median {s["full"].median():.1f}, '
                  f'HMM {s["HMM"].median():.1f}, GLM {s["GLM"].median():.2f}')
        if d['corr'] is not None:
            r = corr_table(d['corr'], 'overall', 'test')
            print(f'  {name}: overall held-out r {r.mean():.2f} +/- {r.std(ddof=0):.2f}')
        fs = d['flicks']
        if fs is not None:
            print(f'  {name}: wing flicks from {fs["folds"]} folds: F1 train {fs["full"]["train_f1"]:.3f}, '
                  f'held-out {fs["full"]["test_f1"]:.3f}; AP {fs["full"]["test_ap"]:.3f} (random {fs["rate"]:.3f}); '
                  f'states only {fs["states"]["test_bits"]:.3f} of {fs["full"]["test_bits"]:.3f} bits/s')
            print('    F1 by state, train: ' + ', '.join(f'{v:.3f}' for v in fs['train_by_state'])
                  + ' | held-out: ' + ', '.join(f'{v:.3f}' for v in fs['test_by_state']))


if __name__ == '__main__':
    config = CONFIG
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config = {**CONFIG, **json.load(f)}
    report(build(config), config['K'])
    print(f'\nWrote figures to {os.path.abspath(config["out"])}')
