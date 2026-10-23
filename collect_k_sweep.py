"""
Collect the K sweep: every fly's held-out score for each number of states K, with
per-fly paired comparisons between consecutive K and each K's share of the best K's score.

Usage:   python collect_k_sweep.py ./models/octladder_cv_wt_female
Needs collect_folds.py in the same folder; like it, reads each model's small scores.json.
Prints a summary to paste back and writes per_fly_by_K.csv.

K is read from 'num_states' in the tuning JSON if present, otherwise from the
checkpoint path ("dghmm_5_cv/...").
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from collect_folds import BITS, MODELS, ensure_scores, get_k, load_checkpoint


N_FOLDS = 10

def paired(d):
    d = np.asarray(d, float)
    return {'median': np.median(d), '% flies > 0': 100 * np.mean(d > 0),
            'Wilcoxon p': wilcoxon(d).pvalue if np.any(d != 0) else np.nan}


def main(results_dir):
    # Latest run for each (K, fold); skip the 'all' fits and older random-split runs.
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '**', '*_tuning.json'), recursive=True),
                       key=os.path.getmtime):
        t = json.load(open(path))
        if isinstance(t.get('fold'), int) and get_k(t) is not None:
            runs[(get_k(t), t['fold'])] = t

    ensure_scores([t['checkpoints'][p] for t in runs.values() for p in MODELS.values()])
    rows = []
    for (k, fold), t in sorted(runs.items()):
        for name, prefix in MODELS.items():
            test_idx, scores = load_checkpoint(t['checkpoints'][prefix])
            test_idx, scores = np.asarray(test_idx).ravel(), np.asarray(scores, float).ravel()
            assert len(test_idx) == len(scores), f'K={k} fold {fold} {name}: length mismatch'
            rows += [{'K': k, 'fold': fold, 'fly': int(f), 'model': name, 'bits/s': s * BITS}
                     for f, s in zip(test_idx, scores)]
    df = pd.DataFrame(rows)
    assert not df.duplicated(['K', 'fly', 'model']).any(), 'a fly was tested twice for the same K'

    folds_per_k = df.groupby('K')['fold'].nunique()
    incomplete = folds_per_k[folds_per_k < N_FOLDS]
    if len(incomplete):
        print('note: incomplete K values (folds found):', incomplete.to_dict(), '- left out below\n')
    ks = sorted(folds_per_k[folds_per_k == N_FOLDS].index)
    df = df[df['K'].isin(ks)]

    # fly x K tables of held-out bits/s over Chance
    full = df[df.model == 'idglmhmm'].pivot(index='fly', columns='K', values='bits/s')
    hmm = df[df.model == 'dghmm'].pivot(index='fly', columns='K', values='bits/s')
    glm = df[(df.model == 'glm') & (df.K == ks[0])].set_index('fly')['bits/s']   # same GLM in every job
    full.insert(0, 1, glm)                                                        # K = 1: single-state GLM
    trans = (df[df.model == 'idglmhmm'].pivot(index='fly', columns='K', values='bits/s')
             - df[df.model == 'lrhmm'].pivot(index='fly', columns='K', values='bits/s'))

    out = []
    for k in full.columns:
        r = {'K': k, 'flies': full[k].notna().sum(),
             'full model (median)': full[k].median(),
             'IQR low': full[k].quantile(.25), 'IQR high': full[k].quantile(.75),
             'HMM no inputs (median)': hmm[k].median() if k in hmm else np.nan,
             'transitions gain (median)': trans[k].median() if k in trans else np.nan}
        prev = [c for c in full.columns if c < k]
        if prev:
            p = paired(full[k] - full[prev[-1]])
            r.update({'gain vs previous K (median)': p['median'], '% flies better': p['% flies > 0'],
                      'Wilcoxon p': p['Wilcoxon p']})
        out.append(r)
    # Diminishing returns: each K's median improvement over Chance as a share of the best K's.
    table = pd.DataFrame(out).set_index('K')
    top = table['full model (median)'].max()
    table.insert(4, '% of best K', 100 * table['full model (median)'] / top)
    pd.set_option('display.width', 240)
    pd.set_option('display.float_format', lambda x: f'{x:.3g}')
    print('Held-out bits/s over Chance by number of states (K = 1 is the single-state GLM):')
    print(table.to_string())

    os.makedirs(f'{results_dir}/csv_results', exist_ok=True)
    df.to_csv(f'{results_dir}/csv_results/per_fly_by_K.csv', index=False)
    print('\nWrote per_fly_by_K.csv')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
