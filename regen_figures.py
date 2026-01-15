####################################
# Script to process, plot post model fitting. Uses saved model.pkl to reload data and configs and plot.
# utils.enhance can be used to get model.pkl provided using model_basic.pkl checkpoint dumped.

# Usage: python regen_figures.py

####################################
import glob
import sys

from utilities import utils


if __name__  == '__main__':

    model_pkl_path = f'../paper figs/FINAL WT/20260101_235805_duration'    # for wt
    # model_pkl_path = f'../paper figs/FINAL WT FRED/20260102_135949_spandex'    # for wt_fred

    # utils.enhance(model_pkl_path)
    # utils.enhance_auxem(model_pkl_path, savefig=1, display=0)

    utils.generate_figures(model_pkl_path, savefig=1, display=0, override_fig_dir=False)
    # utils.generate_state_filters(model_pkl_path, savefig=1, display=0)
    # utils.generate_together_figures(model_pkl_path, savefig=1, display=0)
    # utils.generate_trajs(model_pkl_path, savefig=1, display=0, gen_corr_video=False)

    # utils.enhance_auxem(model_pkl_path, savefig=1, display=0)
    # utils.generate_auxem_figures(model_pkl_path, savefig=1, display=0)

    # utils.generate_state_traces(model_pkl_path, dataset='wt', savefig=1, display=0)

    # utils.generate_TAs(model_pkl_path, savefig=1, display=0)
    #
    # utils.generate_state_clips(model_pkl_path, savefig=1, display=0, gen_corr_video=True)
    # utils.generate_videos(model_pkl_path, override_vid_dir=True)

    # model_dir = 'models/jan1_kfoldcv_wt_female/id-glm-hmm_5_cv/'
    # for path in sorted(list(glob.glob(model_dir + '/20260102_*/'))):
    #     print("path", path)
    #     utils.generate_figures(path, savefig=1, display=0, override_fig_dir=False)
    #     utils.generate_state_filters(path, savefig=1, display=0)

    # utils.generate_figures_all_singles_merged(model_dir, savefig=True, display=True, override_fig_dir=False)

