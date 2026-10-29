####################################
# Script to process, plot post model fitting. Uses saved model.pkl to reload data and configs and plot.
# utils.enhance can be used to get model.pkl provided using model_basic.pkl checkpoint dumped.

# Usage: python regen_figures.py

####################################
import glob
import sys

from utilities import utils


if __name__  == '__main__':

    paths = list(glob.glob('models/octladder_cv_wt_female/id-glm-hmm_5_cv/**/'))
    # paths = list(glob.glob('models/octladder_cv_wt_fred_female/id-glm-hmm_5_cv/**/'))
    print(paths)
    for model_pkl_path in paths:
        # if ('send' not in model_pkl_path):        # wt all-flies fold
        #     continue
        if ('godmother' not in model_pkl_path):     # wt_fred all-flies fold
                continue

        # utils.enhance(model_pkl_path)
        utils.generate_figures(model_pkl_path, savefig=1, display=0, override_fig_dir=False)
        utils.generate_state_filters(model_pkl_path, savefig=1, display=0)

        # utils.enhance_auxem(model_pkl_path)
        utils.generate_together_figures(model_pkl_path, savefig=1, display=0)

        utils.generate_trajs(model_pkl_path, savefig=1, display=0, gen_corr_video=False)
        utils.generate_state_traces(model_pkl_path, dataset='wt', savefig=1, display=0)
        utils.generate_state_clips(model_pkl_path, savefig=1, display=0, gen_corr_video=True)

