"""
Collect the 10-fold results: one row per fly with its held-out score under each model,
paired per-fly comparisons (median, IQR, Wilcoxon), and a per-fold tuning table.

Usage:   python collect_folds.py ./models/octladder_cv_wt_female
Prints a summary to paste back, and writes per_fly_scores.csv in the current folder.
"""
import glob
import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BITS = 30 / np.log(2)   # nats per frame -> bits per second (30 Hz)
N_JOBS = 4              # full checkpoints read in parallel the first time; each holds all the data
 
# name used here -> prefix used as the key in the tuning JSON's 'checkpoints'
MODELS = {'glm': 'lr', 'dghmm': 'dghmm', 'lrhmm': 'lrhmmci_', 'idglmhmm': 'id-glm-hmm'}


def scores_from_checkpoint(ck):
    """The small summary kept in scores.json, built from a loaded model_basic.pkl."""
    tr, te = ck['train_data'], ck['test_data']
    as_list = lambda x: np.asarray(x, float).ravel().tolist()
    return {
        'prefix': ck['prefix'],
        'num_states': int(ck['num_states']),
        'train_session_indices': np.asarray(tr['train_session_indices']).astype(int).tolist(),
        'test_session_indices': np.asarray(te['test_session_indices']).astype(int).tolist(),
        'train_lp': float(tr['train_lp']),
        'test_lp': float(te['test_lp']),
        'train_lps_by_fly': as_list(tr['train_lps_by_fly']),
        'test_lps_by_fly': as_list(te['test_lps_by_fly']),
    }


def write_scores(path):
    """Read one full checkpoint and write scores.json next to it."""
    s = scores_from_checkpoint(joblib.load(os.path.join(path, 'model_basic.pkl')))
    with open(os.path.join(path, 'scores.json'), 'w') as f:
        json.dump(s, f)
    return


def ensure_scores(paths):
    """Write scores.json for every checkpoint folder that doesn't have one yet."""
    missing = sorted({p for p in paths if not os.path.exists(os.path.join(p, 'scores.json'))})
    if not missing:
        return
    print(f'reading {len(missing)} full checkpoints once to write scores.json ...')
    for p in missing:
        if not os.path.exists(os.path.join(p, 'scores.json')):
            write_scores(p)
    return


def get_k(t):
    """Number of states of a run: 'num_states' in its tuning JSON, else from the checkpoint path."""
    if 'num_states' in t:
        return int(t['num_states'])
    m = re.search(r'_(\d+)_cv', str(t['checkpoints'].get(MODELS['dghmm'], '')))
    return int(m.group(1)) if m else None


def load_checkpoint(path):
    """(test session indices, per-fly test scores in nats/frame relative to Chance), same order."""
    with open(os.path.join(path, 'scores.json')) as f:
        s = json.load(f)
    return s['test_session_indices'], s['test_lps_by_fly']


def summarize(d):
    d = np.asarray(d, float)
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    p = wilcoxon(d).pvalue if np.any(d != 0) else np.nan
    return {'n': len(d), 'median': med, 'IQR low': q1, 'IQR high': q3, 'mean': d.mean(),
            '% flies > 0': 100 * np.mean(d > 0), 'Wilcoxon p': p}


def main(results_dir, k=5):
    # One tuning JSON per fold; skip the 'all' fit and the older random-split runs.
    # If a fold was run more than once, keep the most recent run.
    by_fold = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '**', '*_tuning.json'), recursive=True),
                       key=os.path.getmtime):
        t = json.load(open(path))
        if isinstance(t.get('fold'), int) and get_k(t) == k:
            if t['fold'] in by_fold:
                print(f'note: fold {t["fold"]} found more than once; using {path}')
            by_fold[t['fold']] = t
    if not by_fold:
        sys.exit(f'no tuning JSONs for K={k} with an integer fold under {results_dir}')

    ensure_scores([t['checkpoints'][p] for t in by_fold.values() for p in MODELS.values()])
    rows, tuning = [], []
    for fold, t in sorted(by_fold.items()):
        ch = t['chosen']
        r, l1, l2 = ch['emission_l2'], ch['l1'], ch['l2']
        rv, iv = t.get('ridge_val', {}), t.get('l1l2_val', {})
        dg_val, lr_val, id_val = t.get('dghmm_val'), rv.get(f'{r:g}'), iv.get(f'{l1:g},{l2:g}')
        tuning.append({
            'fold': fold, 'ridge': r, 'L2': l2,
            'val lrhmm-dghmm': (lr_val - dg_val) * BITS if None not in (lr_val, dg_val) else np.nan,
            'val id-lrhmm': (id_val - lr_val) * BITS if None not in (id_val, lr_val) else np.nan,
        })
        for name, prefix in MODELS.items():
            test_idx, scores = load_checkpoint(t['checkpoints'][prefix])
            test_idx, scores = np.asarray(test_idx).ravel(), np.asarray(scores, float).ravel()
            assert len(test_idx) == len(scores), f'fold {fold} {name}: {len(scores)} scores for {len(test_idx)} flies'
            rows += [{'fold': fold, 'fly': int(f), 'model': name, 'score': s} for f, s in zip(test_idx, scores)]

    df = pd.DataFrame(rows)
    dup = df.groupby(['fly', 'model']).size()
    assert (dup == 1).all(), f'flies tested more than once: {sorted(set(dup[dup > 1].index.get_level_values(0)))}'
    wide = df.pivot(index='fly', columns='model', values='score')
    wide.insert(0, 'fold', df.groupby('fly')['fold'].first())
    print(f'{len(wide)} flies from {len(by_fold)} folds: {sorted(by_fold)}\n')

    # Per-fly scores in bits/s (vs Chance), and the paired differences.
    for m in MODELS:
        wide[f'{m} (bits/s)'] = wide[m] * BITS
    comparisons = {
        'id-glm-hmm - lrhmm (cue transitions)': wide['idglmhmm'] - wide['lrhmm'],
        'lrhmm - dghmm': wide['lrhmm'] - wide['dghmm'],
        'id-glm-hmm - dghmm': wide['idglmhmm'] - wide['dghmm'],
        'glm - chance': wide['glm'],
        'dghmm - chance': wide['dghmm'],
        'id-glm-hmm - chance': wide['idglmhmm'],
    }
    for label, d in comparisons.items():
        wide[label] = d * BITS
    os.makedirs(f'{results_dir}', exist_ok=True)
    wide.to_csv(f'{results_dir}/per_fly_scores_k={k}.csv')

    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: f'{x:.3g}')
    print('Held-out flies, bits/s (per-fly paired differences):')
    print(pd.DataFrame({k: summarize(v * BITS) for k, v in comparisons.items()}).T.to_string(), '\n')

    per_fold = wide.groupby('fold').agg(
        flies=('dghmm', 'size'),
        **{'test id-lrhmm (median)': ('id-glm-hmm - lrhmm (cue transitions)', 'median'),
           'test lrhmm-dghmm (median)': ('lrhmm - dghmm', 'median')})
    print('Per fold (validation gains from tuning, test medians over that fold\'s flies), bits/s:')
    print(pd.DataFrame(tuning).set_index('fold').join(per_fold).to_string())
    print('\nWrote per_fly_scores.csv')
    return


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.', int(sys.argv[2]) if len(sys.argv) > 2 else 5)

