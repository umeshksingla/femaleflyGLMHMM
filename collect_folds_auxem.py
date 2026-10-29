import glob, json, os, joblib, sys
from collect_folds import get_k
from utilities.flick_model import pool_folds
from pprint import pprint


def main(results_dir, k=5):
    runs = {}
    for p in sorted(glob.glob(os.path.join(results_dir, '**', '*_tuning.json'), recursive=True),
                    key=os.path.getmtime):
        t = json.load(open(p))
        if isinstance(t.get('fold'), int) and get_k(t) == k:
            runs[t['fold']] = t                                   # latest run of each fold
    res = [joblib.load(os.path.join(t['checkpoints']['id-glm-hmm'], 'auxem_model.pkl')) for t in runs.values()]
    pprint(pool_folds(res, o=0, fps=30))
    return


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.', int(sys.argv[2]) if len(sys.argv) > 2 else 5)

