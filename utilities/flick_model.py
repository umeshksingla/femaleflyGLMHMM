"""
Post-hoc wing-flick model. Within each latent state, a logistic regression maps the male cues to
whether the female flicks; the states are combined by their probabilities:

    p(flick_t) = sum_k p(z_t = k) * sigmoid(w_k . x_t + b_k)

Used in two ways:
  - flick_performance (cross-validation folds): fit on the fold's training flies and evaluate on
    its test flies, using one-step-ahead state probabilities p(z_t | y_1..t-1) for both, so the
    state used at frame t is inferred only from earlier frames.
  - flick_filters (the fit on all flies): fit on all flies with smoothed state probabilities, for
    the filter figures; no evaluation.

The regularization strength C is chosen by grouped cross-validation over training flies (held-out
log-likelihood), and the F1 decision threshold is chosen on training flies; test flies are used
only for evaluation. Probabilities are left calibrated (no class re-weighting), so they can be read
as flick rates and scored by log-likelihood against a constant-rate model.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold

# C values tried (inverse ridge strength), narrowed around the 1e-6 that the folds and the fit on
# all flies chose. If a fold picks 1e-4 (the top value), add 1e-3 and rerun that fold.
# Picking the bottom value means the cues add little beyond each state's flick rate.
CS = 10.0 ** np.arange(-7, -3)
MIN_WEIGHT = 1e-3               # frames less likely than this to be in a state are left out of its fit
INNER_SPLITS = 3                # groups of training flies used to choose C


def _stack(per_fly):
    return np.concatenate([np.asarray(a, float) for a in per_fly], axis=0)


def _fly_ids(per_fly):
    return np.concatenate([np.full(len(a), i) for i, a in enumerate(per_fly)])


def fit_states(X, y, P, C):
    """One logistic regression per state, each frame weighted by its probability of that state.
    X (T, D) inputs for this output, y (T,) 0/1 flicks, P (T, K) state probabilities."""
    models = []
    for k in range(P.shape[1]):
        m = P[:, k] > MIN_WEIGHT
        if not m.any() or y[m].min() == y[m].max():   # state never used, or no flicks in it
            rate = np.average(y[m], weights=P[m, k]) if m.any() else y.mean()
            models.append(('rate', float(rate)))
            continue
        lr = LogisticRegression(C=C, max_iter=1000)
        lr.fit(X[m], y[m], sample_weight=P[m, k])
        models.append(('logreg', lr))
    return models


def predict(models, X, P):
    """p(flick_t) = sum_k P[t, k] * p_k(flick_t | x_t)."""
    p = np.zeros(len(X))
    for k, (kind, m) in enumerate(models):
        p += P[:, k] * (m if kind == 'rate' else m.predict_proba(X)[:, 1])
    return np.clip(p, 1e-9, 1 - 1e-9)


def log_lik(y, p):
    return float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))


def choose_C(X, y, P, flies):
    """C with the best held-out log-likelihood over groups of training flies."""
    n_splits = min(INNER_SPLITS, len(np.unique(flies)))
    total = {C: 0.0 for C in CS}
    for tr, va in GroupKFold(n_splits).split(X, y, flies):
        for C in CS:
            total[C] += log_lik(y[va], predict(fit_states(X[tr], y[tr], P[tr], C), X[va], P[va]))
    return float(max(total, key=total.get))


def best_threshold(y, p):
    """Decision threshold that maximizes F1 on these (training) frames."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[np.argmax(f1)])


def metrics(y, p, threshold, base_rate, fps):
    """F1 at the training threshold, plus two threshold-free measures: average precision, and the
    log-likelihood gain over a constant flick rate (the training rate), in bits/s."""
    pred = p >= threshold
    tp, fp, fn = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 1)))
    gain = log_lik(y, p) - log_lik(y, np.full(len(y), np.clip(base_rate, 1e-9, 1 - 1e-9)))
    return {'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'tp': tp, 'fp': fp, 'fn': fn,
            'average_precision': float(average_precision_score(y, p)) if y.any() else np.nan,
            'll_gain_nats': gain, 'frames': len(y), 'flicks': int(y.sum()),
            'bits_per_s_over_rate': float(gain / len(y) / np.log(2) * fps)}


def _weights(models, cols, D):
    """Filters in the full input space (zeros outside this output's inputs) and biases."""
    w, b = np.zeros((len(models), D)), np.zeros(len(models))
    for k, (kind, m) in enumerate(models):
        if kind == 'logreg':
            w[k, cols], b[k] = m.coef_[0], m.intercept_[0]
        else:
            b[k] = np.log(m / (1 - m)) if 0 < m < 1 else np.nan
    return w, b


def flick_performance(train, test, mask, fps):
    """One cross-validation fold. train/test: dicts of per-fly lists 'inputs' (T, D),
    'flicks' (T, n_out) and 'probs' (T, K, one-step-ahead state probabilities).
    Returns {output: results}, with F1 etc. on training and test flies and the test predictions."""
    Xtr_all, Xte_all = _stack(train['inputs']), _stack(test['inputs'])
    Ytr, Yte = _stack(train['flicks']).astype(int), _stack(test['flicks']).astype(int)
    Ptr, Pte = _stack(train['probs']), _stack(test['probs'])
    flies, fly_te = _fly_ids(train['inputs']), _fly_ids(test['inputs'])
    out = {}
    for o in range(Ytr.shape[1]):
        cols = np.asarray(mask[o]) == 1
        Xtr, Xte, ytr, yte = Xtr_all[:, cols], Xte_all[:, cols], Ytr[:, o], Yte[:, o]
        C = choose_C(Xtr, ytr, Ptr, flies)
        models = fit_states(Xtr, ytr, Ptr, C)
        ptr, pte = predict(models, Xtr, Ptr), predict(models, Xte, Pte)
        thr, rate = best_threshold(ytr, ptr), float(ytr.mean())
        w, b = _weights(models, cols, Xtr_all.shape[1])
        state = Pte.argmax(1)
        # States-only baseline: each state's flick rate on the training flies, no cues.
        # The full model's advantage over it is what the cues add beyond the state.
        rates = (Ptr * ytr[:, None]).sum(0) / np.maximum(Ptr.sum(0), 1e-12)
        s_tr, s_te = np.clip(Ptr @ rates, 1e-9, 1 - 1e-9), np.clip(Pte @ rates, 1e-9, 1 - 1e-9)
        s_thr = best_threshold(ytr, s_tr)
        out[o] = {'C': C, 'threshold': thr, 'w': w, 'b': b,
                  'train': metrics(ytr, ptr, thr, rate, fps),
                  'test': metrics(yte, pte, thr, rate, fps),
                  'test_by_state': {k: metrics(yte[state == k], pte[state == k], thr, rate, fps)
                                    for k in range(Pte.shape[1]) if np.any(state == k)},
                  'states_only': {'rates': rates, 'threshold': s_thr,
                                  'train': metrics(ytr, s_tr, s_thr, rate, fps),
                                  'test': metrics(yte, s_te, s_thr, rate, fps)},
                  'test_probs': pte, 'states_only_test_probs': s_te, 'test_true': yte, 'test_fly': fly_te}
    return out


def flick_filters(inputs, flicks, probs, mask):
    """Filters for figures, fit on all flies with smoothed state probabilities."""
    X_all, Y, P, flies = _stack(inputs), _stack(flicks).astype(int), _stack(probs), _fly_ids(inputs)
    K, n_out, D = P.shape[1], Y.shape[1], X_all.shape[1]
    w, b, chosen = np.zeros((K, n_out, D)), np.zeros((K, n_out)), {}
    for o in range(n_out):
        cols = np.asarray(mask[o]) == 1
        chosen[o] = choose_C(X_all[:, cols], Y[:, o], P, flies)
        w[:, o], b[:, o] = _weights(fit_states(X_all[:, cols], Y[:, o], P, chosen[o]), cols, D)
    return {'w': w, 'b': b, 'C': chosen}


def pool_folds(fold_results, o=0, fps=30):
    """Held-out performance pooled over folds (each fly once), for the full model (states + cues)
    and the states-only baseline, plus a per-fly test of what the cues add beyond the state."""
    from scipy.stats import wilcoxon
    y = np.concatenate([r[o]['test_true'] for r in fold_results])
    out = {'test_flick_rate': float(y.mean()),           # average precision of a random guess
           'chosen_C': [r[o]['C'] for r in fold_results]}
    for name, get, probs in (('full', lambda r: r[o], 'test_probs'),
                             ('states_only', lambda r: r[o]['states_only'], 'states_only_test_probs')):
        te = [get(r)['test'] for r in fold_results]
        tp, fp, fn = (sum(m[k] for m in te) for k in ('tp', 'fp', 'fn'))
        p = np.concatenate([r[o][probs] for r in fold_results])
        out[name] = {'test_f1': 2 * tp / max(2 * tp + fp + fn, 1),
                     'test_average_precision': float(average_precision_score(y, p)),
                     'test_bits_per_s_over_rate': float(sum(m['ll_gain_nats'] for m in te)
                                                        / sum(m['frames'] for m in te) / np.log(2) * fps),
                     'train_f1_mean': float(np.mean([get(r)['train']['f1'] for r in fold_results]))}
    # Per held-out fly: log-likelihood gain of the full model over the states-only baseline, in bits/s.
    gains = []
    for r in fold_results:
        rr = r[o]
        for f in np.unique(rr['test_fly']):
            m = rr['test_fly'] == f
            yf = rr['test_true'][m]
            gains.append((log_lik(yf, rr['test_probs'][m]) - log_lik(yf, rr['states_only_test_probs'][m]))
                         / m.sum() / np.log(2) * fps)
    gains = np.array(gains)
    out['cues_beyond_states'] = {'flies': len(gains), 'median_bits_per_s': float(np.median(gains)),
                                 'flies_better': int(np.sum(gains > 0)),
                                 'wilcoxon_p': float(wilcoxon(gains).pvalue) if np.any(gains != 0) else np.nan}
    return out

