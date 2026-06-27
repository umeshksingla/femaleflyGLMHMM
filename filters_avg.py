import glob
import joblib
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


# model_pkl_paths = sorted(glob.glob(f'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/**/'))

dataset = 'wt'
model_pkl_paths = ['models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260531_183347_obsidian/', 'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260624_043651_terminology/', 'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260624_042907_tablecloth/', 'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260624_045656_ethnicity/', 'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260624_045905_downforce/', 'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260531_184413_grin/', 'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260531_202540_drawer/', 'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260624_045656_gastronomy/', 'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260624_045659_mule/', 'models/may31l1l2_sweepcv_wt_female/id-glm-hmm_5_cv/20260624_045453_malice/']

# dataset = 'wt_fred'
# model_pkl_paths = ['models/june26l1l2_sweepcv_wt_fred_female/id-glm-hmm_5_cv/20260626_050049_whelp/', 'models/june26l1l2_sweepcv_wt_fred_female/id-glm-hmm_5_cv/20260626_072922_identification/', 'models/june26l1l2_sweepcv_wt_fred_female/id-glm-hmm_5_cv/20260626_044632_anything/', 'models/june26l1l2_sweepcv_wt_fred_female/id-glm-hmm_5_cv/20260626_045715_crew/', 'models/june26l1l2_sweepcv_wt_fred_female/id-glm-hmm_5_cv/20260626_070408_legitimacy/', 'models/june26l1l2_sweepcv_wt_fred_female/id-glm-hmm_5_cv/20260626_071142_vitality/', 'models/june26l1l2_sweepcv_wt_fred_female/id-glm-hmm_5_cv/20260626_045209_spite/', 'models/june26l1l2_sweepcv_wt_fred_female/id-glm-hmm_5_cv/20260626_045614_summarize/', 'models/june26l1l2_sweepcv_wt_fred_female/id-glm-hmm_5_cv/20260626_044429_sundae/']

all_weights = load_weights(model_pkl_paths)
joblib.dump(all_weights, f'all_weights_{dataset}.pkl')
all_tr_weights = joblib.load(f'all_weights_{dataset}.pkl')
avg_weight = np.mean(all_weights, axis=0)
print(all_weights.shape, "avg_weight.shape", avg_weight.shape)
utils.generate_together_figures_filters_given(model_pkl_paths[0], avg_weight, savefig=True, display=False)

all_tr_weights = load_tr_weights(model_pkl_paths)
joblib.dump(all_tr_weights, f'all_tr_weights_{dataset}.pkl')
all_tr_weights = joblib.load(f'all_tr_weights_{dataset}.pkl')
avg_tr_weight = np.mean(all_tr_weights, axis=0)
print(all_tr_weights.shape, "avg_tr_weight.shape", avg_tr_weight.shape)
utils.generate_state_filters_filters_given(model_pkl_paths[0], avg_tr_weight, savefig=True, display=False)
