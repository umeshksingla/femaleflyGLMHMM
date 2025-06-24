####################################

# Usage: python regen_figures.py

####################################
import glob
import joblib
import os

from utilities import utils


if __name__  == '__main__':

    model_pkl_path = f'models/general_wt/GLM-HMM_2_cv/20250624_020336_revitalisation'
    # utils.enhance(model_pkl_path)
    utils.generate_figures(model_pkl_path, savefig=1, display=0, override_fig_dir=False)
    utils.generate_trajs(model_pkl_path, savefig=1, display=0, gen_corr_video=False)
    # utils.generate_videos(model_pkl_path, override_vid_dir=True)

    # model_dir = 'models/general_cop/lrhmmci2_4/20250508_185109_fragrance'

    # for _ in glob.glob(model_dir + '/session*'):
    #     utils.generate_figures_single(_)

    # utils.generate_figures_all_singles_merged(model_dir, savefig=True, display=True, override_fig_dir=False)

