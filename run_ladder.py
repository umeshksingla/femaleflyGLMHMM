"""
One SLURM job per data split.

  A. Tune on the training sessions only: split them into inner-train / validation,
     fit dghmm (best of init_seeds) -> lrhmm (one per ridge value) -> id-glm-hmm (one per (l1, l2)),
     and choose the penalties by validation log-likelihood.
  B. Refit the full ladder on all training sessions with the chosen penalties.
  C. Save every rung (chance, lr, dghmm, lrhmm, id-glm-hmm) as its own checkpoint, plus the tuning table.
     Test scoring happens in utils.enhance, as before.

Called from run_global.py when mc['name'] == 'ladder'.
Needs these keys in mc (besides the usual ones): init_seeds, emission_l2_grid, l1_grid, l2_grid.
"""
import itertools
import json

import joblib
import numpy as np

from hmms.ChanceFemaleFly import ChanceFemaleFly
from hmms.LRFemaleFly import LRFemaleFly
from hmms.DiagGHMMFemaleFly import DiagGHMMFemaleFly
from hmms.LRHMMCustomInitFemaleFly import LRHMMCustomInitFemaleFly
from hmms.InputDrivenLRHMMCustomInitFemaleFly import InputDrivenLRHMMCustomInitFemaleFly
from utilities import utils


def _with(mc, **overrides):
    """Copy of the model config with some keys replaced."""
    c = dict(mc)
    c.update(overrides)
    return c


def _subset(idx, emissions, inputs, mn_std):
    return [emissions[i] for i in idx], [inputs[i] for i in idx], [mn_std[i] for i in idx]


def _score(model, emissions, inputs):
    """Log-likelihood per frame relative to Chance; higher is better.
    Adjust if your HMM classes expose this under a different name."""
    ll = float(model.get_data_logprob(emissions, inputs))
    return ll if np.isfinite(ll) else -np.inf


def _best_dghmm(data_config, mc, e, i, m):
    """Fit the diagonal HMM from every init seed and keep the best by TRAINING log-likelihood."""
    best, best_ll = None, -np.inf
    for j, s in enumerate(mc['init_seeds']):
        init_method = 'kmeans' if j == 0 else 'frames'
        print(f'dghmm: init seed={s} init_method={init_method}')
        dg = DiagGHMMFemaleFly(data_config, _with(mc, name='dghmm', seed=s, init_method=init_method))
        dg.fit(e, i, m, do_reindex_params=False)
        ll = _score(dg, e, i)
        print(f'  dghmm seed {s}: train {ll:.5f}')
        if ll > best_ll:
            best, best_ll = dg, ll
    print(f'dghmm: BEST train {best_ll:.5f}')
    return best


def _fit_lrhmm(data_config, mc, ridge, dg, e, i, m):
    lr = LRHMMCustomInitFemaleFly(data_config, _with(mc, name='lrhmmci', emission_l2_penalty=ridge))
    lr.fit(e, i, m, dghmm=dg, do_reindex_params=False)          # starts exactly at dghmm
    return lr


def _fit_idglmhmm(data_config, mc, ridge, l1, l2, lr, e, i, m):
    idm = InputDrivenLRHMMCustomInitFemaleFly(data_config,
                                              _with(mc, name='idglmhmmci', emission_l2_penalty=ridge, l1_penalty=l1, l2_penalty=l2))
    idm.fit(e, i, m, lrhmmci=lr, do_reindex_params=False)         # starts exactly at lrhmmci
    return idm


def run_ladder(mc):
    data = joblib.load(mc['data_path'])
    emissions, inputs, output_mn_std = data['emissions'], data['inputs'], data['output_mn_std']
    data_config = data['data_config']

    # Outer train/test split: identical to run_global.py, so results stay paired with earlier runs.
    num_batches = data_config['num_sessions']
    print('total num_batches', num_batches)
    fold = mc['fold']
    n_folds = mc.get('n_folds', 10)
    perm = np.random.default_rng(1326).permutation(num_batches)    # same seed in every job

    if fold == 'all':
        train_idx, test_idx = np.arange(num_batches), np.array([], dtype=int)
        inner_seed = 999
    else:
        test_idx = np.sort(np.array_split(perm, n_folds)[fold])
        train_idx = np.setdiff1d(perm, test_idx)
        inner_seed = 1000 + fold

    # Inner split of the TRAINING sessions only; the test sessions are never touched while tuning.
    shuffled = np.random.default_rng(inner_seed).permutation(train_idx)
    n_inner = int(len(shuffled) * 0.8)
    inner_idx, val_idx = shuffled[:n_inner], shuffled[n_inner:]

    tr = _subset(train_idx, emissions, inputs, output_mn_std)
    inner_tr = _subset(inner_idx, emissions, inputs, output_mn_std)
    val = _subset(val_idx, emissions, inputs, output_mn_std)
    print('# Train:', train_idx, '# Test:', test_idx)
    print('# Inner-train:', inner_idx, '# Validation:', val_idx)

    # ---------------- A. Tuning: fit on inner-train, score on validation ----------------
    if mc.get('tune', True):
        print('\n>> TUNING')
        dg = _best_dghmm(data_config, mc, *inner_tr)
        dg_val = _score(dg, val[0], val[1])
        print(f'  dghmm: val {dg_val:.5f}')

        ridge_val = {}
        lrhmm_fits = {}
        for r in mc['emission_l2_grid']:
            lrhmm_fits[r] = _fit_lrhmm(data_config, mc, r, dg, *inner_tr)
            ridge_val[r] = _score(lrhmm_fits[r], val[0], val[1])
            print(f'  lrhmm ridge {r:g}: val {ridge_val[r]:.5f}')
        best_r = max(ridge_val, key=ridge_val.get)

        # Now we have emmission hyperparameters from validation data
        print(f'>> Chosen: ridge {best_r:g}')

        l1l2_val = {}
        for l1, l2 in itertools.product(mc['l1_grid'], mc['l2_grid']):
            idm = _fit_idglmhmm(data_config, mc, best_r, l1, l2, lrhmm_fits[best_r], *inner_tr)
            l1l2_val[(l1, l2)] = _score(idm, val[0], val[1])
            print(f'  id-glm-hmm l1 {l1:g}, l2 {l2:g}: val {l1l2_val[(l1, l2)]:.5f}')
        best_l1, best_l2 = max(l1l2_val, key=l1l2_val.get)

        # Now we have all hyperparameters from validation data
        print(f'>> Chosen: ridge {best_r:g}, l1 {best_l1:g}, l2 {best_l2:g}')
    else:
        print('\n>> NO TUNING NEEDED')
        best_r, best_l1, best_l2 = mc['emission_l2_penalty'], mc['l1_penalty'], mc['l2_penalty']
        ridge_val, l1l2_val, dg_val = {}, {}, ""    # so the tuning JSON still writes

    # ---------------- B. Final ladder on all training sessions ----------------
    print('\n\n\n>> FINAL FITS')
    chance = ChanceFemaleFly(data_config, _with(mc, name='chance', num_states=1))
    chance.fit(*tr)
    glm = LRFemaleFly(data_config, _with(mc, name='lr', num_states=1))
    glm.fit(*tr)
    dghmm = _best_dghmm(data_config, mc, *tr)
    lrhmmci = _fit_lrhmm(data_config, mc, best_r, dghmm, *tr)
    idglmhmmci = _fit_idglmhmm(data_config, mc, best_r, best_l1, best_l2, lrhmmci, *tr)

    # Order states by speed only now, after all handoffs are done.
    for model in (dghmm, lrhmmci, idglmhmmci):
        model.learned_params = model.reindex_params(model.learned_params, *tr)

    # ---------------- C. Save every rung, plus the tuning table ----------------
    paths = {}
    for model in (chance, glm, dghmm, lrhmmci, idglmhmmci):
        path = utils.getafilepath(f'{mc["path"]}/{model.prefix}_{model.model_config["num_states"]}_cv')
        utils.save(model, data, train_idx, test_idx, path)
        paths[model.prefix] = path
        print('>> Saved', model.prefix, 'at', path)

    tuning = {
        'fold': fold,
        'chosen': {'emission_l2': best_r, 'l1': best_l1, 'l2': best_l2},
        'dghmm_val': dg_val,
        'ridge_val': {f'{k:g}': v for k, v in ridge_val.items()},
        'l1l2_val': {f'{k[0]:g},{k[1]:g}': v for k, v in l1l2_val.items()},
        'inner_train_sessions': inner_idx.tolist(),
        'val_sessions': val_idx.tolist(),
        'checkpoints': {k: str(v) for k, v in paths.items()},
    }
    print(tuning)
    with open(f'{paths[idglmhmmci.prefix]}_tuning.json', 'w') as f:
        json.dump(tuning, f, indent=2)
    return paths
