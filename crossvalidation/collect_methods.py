import glob, json
import numpy as np
import sys
from collect_folds import get_k

import pandas as pd
from scipy.stats import wilcoxon

B = 30 / np.log(2)

def main(results_dir):

    # 1. Test of the cue-dependent transitions gain
    df = pd.read_csv(f'{results_dir}/per_fly_by_K.csv')
    d = df.pivot_table(index=['K', 'fly'], columns='model', values='bits/s')
    gain = d['idglmhmm'] - d['lrhmm']
    for k, g in gain.groupby(level='K'):
        print(f'K={k}: median {g.median():.3f}, {(g > 0).sum()} of {len(g)} flies better, p = {wilcoxon(g).pvalue:.2g}')

    print()
    # 2. A reference model. ID-GLMHMM with very high regularization on transitions (l2=1e8), making it close to being simply an LRHMM.
    # How does it behave compared to ID-GLMHMM with chosen regularization on validation?
    # It's to disambiguate some of the folds where it seemed like LRHMM could have gone further in EM, so whether ID_GLMHMM
    # is better just coz of more EM iterations or genuinely coz of added cues on transitions.
    gaps = []
    for p in sorted(glob.glob(f'{results_dir}/**/*_tuning.json', recursive=True)):
        t = json.load(open(p))
        if isinstance(t.get('fold'), int) and get_k(t) == 5:
            r, l1, l2 = t['chosen']['emission_l2'], t['chosen']['l1'], t['chosen']['l2']
            dghmm = t['dghmm_val']
            lrhmm = t['ridge_val'][f'{r:g}']
            ref = (t['l1l2_val']['1,1e+08'] - lrhmm) * B             # cues on emissions but effectively no cue-dependent transitions
            chosen = (t['l1l2_val'][f'{l1:g},{l2:g}'] - lrhmm) * B   # the selected L2
            gaps.append(ref)
            print(f"fold {t['fold']}:\treference (l1=1,l2=1e+08) {ref:+.3f},\tchosen (l1={l1:g},l2={l2:g}) {chosen:+.3f},\tso tr cues added {chosen - ref:+.3f} bits/s")
    print(f'{len(gaps)} folds, largest gap {max(gaps):.3f} bits/s')
    
    return


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
