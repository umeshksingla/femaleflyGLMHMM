import glob
import numpy as np

from utilities import io, utils


def load_weights(paths):
    glm_weights = []
    for i, p in enumerate(paths):
        print(p)
        pkl, _, _ = io.load_specific_path(p)
        auxem_model_ckp = io.load_specific_path_auxem(p)
        if pkl is None:
            continue
        reg_weights = pkl['learned_params'].emissions.weights
        aux_weights = auxem_model_ckp['logreg_params']['w']
        w = np.concatenate((reg_weights, aux_weights), axis=1)
        print(w.shape)
        glm_weights.append(w)
        # if i == 1:
        #     break
    return np.array(glm_weights)


def load_tr_weights(paths):
    tr_glm_weights = []
    for i, p in enumerate(paths):
        print(p)
        pkl, _, _ = io.load_specific_path(p)
        auxem_model_ckp = io.load_specific_path_auxem(p)
        if pkl is None:
            continue
        tr_weights = pkl['learned_params'].transitions.weights
        w = tr_weights
        print(w.shape)
        tr_glm_weights.append(w)
        # if i == 1: break
    return np.array(tr_glm_weights)


model_pkl_paths = sorted(glob.glob(f'models/apr6_bothcv_wt_female_2/id-glm-hmm_5_cv/**/'))

# all_weights = load_weights(model_pkl_paths)
# avg_weight = np.mean(all_weights, axis=0)
# print(all_weights.shape, "avg_weight.shape", avg_weight.shape)
# utils.generate_together_figures_filters_given(model_pkl_paths[0], avg_weight, savefig=True, display=False)

all_tr_weights = load_tr_weights(model_pkl_paths)
avg_tr_weight = np.mean(all_tr_weights, axis=0)
print(all_tr_weights.shape, "avg_tr_weight.shape", avg_tr_weight.shape)
utils.generate_state_filters_filters_given(model_pkl_paths[0], avg_tr_weight, savefig=True, display=False)
