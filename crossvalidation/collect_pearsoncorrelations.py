"""
Pool the per-fly correlations that utils.enhance stored in each fold's model.pkl.
  - Test: each fly's value from the fold that held it out (each fly once).
  - Train: each fly's values averaged over the folds in which it was a training fly.
Variance explained is reported as r^2, averaged over flies.

Usage:   python collect_correlations.py ./models/octladder_cv_wt_female [K] [prefix]
         K defaults to 5; prefix defaults to id-glm-hmm (use lr for the GLM, dghmm, lrhmmci_ ...).
Prints a summary and writes correlations_by_fly.csv (one row per fly, split, measure, state, output).

The first run reads each model.pkl once (N_JOBS at a time, slow) and writes a small
correlations.json next to it, so later runs are quick. Delete those files if you re-enhance.
"""
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from collect_folds import N_JOBS, get_k

# name used here -> key in model.pkl's train_data / test_data, after the 'train_' / 'test_' prefix
KEYS = {
    'overall': 'pearson_by_fly',
    'by_output': 'correlation_by_o_by_fly_soft',
    'by_state': 'pearson_by_z_by_fly',
    # 'by_state_and_output': 'pearson_by_z_and_o_by_fly',
}
OUTPUTS = ['forward', 'lateral', 'angular']


def _as_array(value):
    """Per-fly values as a float array with flies on the first axis, or None if not rectangular.
    Dicts keyed by output or state (values: per-fly arrays, or further dicts) are stacked in key
    order after the fly axis, e.g. {output: (flies,)} -> (flies, outputs) and
    {state: {output: (flies,)}} -> (flies, states, outputs)."""
    if value is None:
        return None
    if isinstance(value, dict):
        parts = [_as_array(value[k]) for k in sorted(value)]
        if not parts or any(p is None for p in parts) or len({p.shape for p in parts}) != 1:
            return None
        return np.stack(parts, axis=1)
    try:
        return np.asarray([np.asarray(v, float) for v in value], float)
    except (TypeError, ValueError):
        return None


def extract(path):
    """Read one enhanced checkpoint and cache every stored correlation in correlations.json."""
    ck = joblib.load(os.path.join(path, 'model.pkl'))
    out = {}
    for split in ('train', 'test'):
        d = ck[f'{split}_data']
        out[f'{split}_session_indices'] = np.asarray(d[f'{split}_session_indices']).astype(int).tolist()
        vals = {}
        for key, value in d.items():
            name = key[len(split) + 1:] if key.startswith(f'{split}_') else key
            if not any(s in name for s in ('pearson', 'correlation')):
                continue
            arr = _as_array(value)
            if arr is not None:
                vals[name] = np.where(np.isfinite(arr), arr, None).tolist()
        out[split] = vals
    with open(os.path.join(path, 'correlations.json'), 'w') as f:
        json.dump(out, f)


def load(path):
    with open(os.path.join(path, 'correlations.json')) as f:
        c = json.load(f)
    for split in ('train', 'test'):
        c[split] = {k: (None if v is None else np.array(v, dtype=float)) for k, v in c[split].items()}
    return c


def per_fly(r, split, key):
    """The stored array if it has one entry per fly of this split, else None."""
    v = r[split].get(key)
    return v if v is not None and v.ndim >= 1 and len(v) == len(r[f'{split}_session_indices']) else None


def pool(runs, n_flies=None):
    """runs: list of loaded correlations.json dicts, one per fold.
    Returns {measure: {'test': (flies, ...), 'train': (flies, ...)}} indexed by session."""
    if n_flies is None:
        n_flies = 1 + max(max(r['test_session_indices'] + r['train_session_indices']) for r in runs)
    pooled = {}
    for name, key in KEYS.items():
        if any(per_fly(r, s, key) is None for r in runs for s in ('train', 'test')):
            shapes = sorted({str(np.shape(r['test'].get(key))) for r in runs})
            print(f'  skipped {name} ({key}): not one entry per fly (test shapes {", ".join(shapes)}); '
                  f'see the list above and set KEYS')
            continue
        shape = runs[0]['test'][key].shape[1:]
        test = np.full((n_flies,) + shape, np.nan)
        tsum, tcount = np.zeros((n_flies,) + shape), np.zeros((n_flies,) + shape)
        for r in runs:
            test[r['test_session_indices']] = r['test'][key]
            v, idx = r['train'][key], np.asarray(r['train_session_indices'])
            np.add.at(tsum, idx, np.nan_to_num(v))
            np.add.at(tcount, idx, np.isfinite(v))
        pooled[name] = {'test': test, 'train': np.where(tcount > 0, tsum / np.maximum(tcount, 1), np.nan)}
    return pooled


def _labels(name, shape):
    """(state, output) label for each column of a per-fly array with this trailing shape."""
    out = lambda j, n: OUTPUTS[j] if n == len(OUTPUTS) else str(j)
    if not shape:
        return [(None, None)]
    if name == 'by_state':
        return [(k + 1, None) for k in range(shape[0])]
    if len(shape) == 1:
        return [(None, out(j, shape[0])) for j in range(shape[0])]
    if len(shape) == 2:
        return [(k + 1, out(j, shape[1])) for k in range(shape[0]) for j in range(shape[1])]
    return [(None, str(j)) for j in range(int(np.prod(shape)))]


def summarize(pooled):
    rows = []
    for name, d in pooled.items():
        shape = d['test'].shape[1:]
        flat = {s: d[s].reshape(len(d[s]), -1) for s in ('test', 'train')}
        for j, (state, output) in enumerate(_labels(name, shape)):
            te, tr = flat['test'][:, j], flat['train'][:, j]
            rows.append({'measure': name, 'state': state, 'output': output,
                         'flies': int(np.isfinite(te).sum()),
                         'test r (mean)': np.nanmean(te), 'test r (SD)': np.nanstd(te),
                         'test r (median)': np.nanmedian(te),
                         'test r^2 (mean, %)': 100 * np.nanmean(te ** 2),
                         'train r (mean)': np.nanmean(tr), 'train r (SD)': np.nanstd(tr)})
    return pd.DataFrame(rows)


def to_long(pooled, fold_of):
    rows = []
    for name, d in pooled.items():
        labels = _labels(name, d['test'].shape[1:])
        for split in ('test', 'train'):
            flat = d[split].reshape(len(d[split]), -1)
            for fly in range(len(flat)):
                for j, (state, output) in enumerate(labels):
                    rows.append({'fly': fly, 'fold': fold_of.get(fly), 'split': split, 'measure': name,
                                 'state': state, 'output': output, 'r': flat[fly, j]})
    return pd.DataFrame(rows)


def main(results_dir, k=5, prefix='id-glm-hmm'):
    runs = {}
    for p in sorted(glob.glob(os.path.join(results_dir, '**', '*_tuning.json'), recursive=True),
                    key=os.path.getmtime):
        t = json.load(open(p))
        if isinstance(t.get('fold'), int) and get_k(t) == k:
            runs[t['fold']] = t['checkpoints'][prefix]          # latest run of each fold
    if not runs:
        sys.exit(f'no K={k} folds under {results_dir}')
    missing = [p for p in runs.values() if not os.path.exists(os.path.join(p, 'correlations.json'))]
    if missing:
        print(f'reading {len(missing)} enhanced checkpoints once to write correlations.json ...')
        for p in missing:
            extract(p)
    loaded = {fold: load(p) for fold, p in sorted(runs.items())}
    fold_of = {fly: fold for fold, c in loaded.items() for fly in c['test_session_indices']}
    print(f'{prefix}, K={k}: {len(loaded)} folds, {len(fold_of)} held-out flies')
    first_fold, first = next(iter(loaded.items()))
    print(f'Stored correlations in fold {first_fold} ({len(first["test_session_indices"])} test flies); '
          f'per-fly ones start with that number:')
    for name, v in sorted(first['test'].items()):
        print(f'  {name}: shape {np.shape(v)}')
    pooled = pool(list(loaded.values()))
    for name, d in pooled.items():
        print(f'  pooled {name}: per-fly array shape {d["test"].shape}')
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', lambda x: f'{x:.3g}')
    print(summarize(pooled).to_string(index=False))
    os.makedirs(f'{results_dir}', exist_ok=True)
    to_long(pooled, fold_of).to_csv(f'{results_dir}/correlations_by_fly_{prefix}_k={k}.csv', index=False)
    print('\nWrote correlations_by_fly.csv')
    return


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.',
         int(sys.argv[2]) if len(sys.argv) > 2 else 5,
         sys.argv[3] if len(sys.argv) > 3 else 'id-glm-hmm')
