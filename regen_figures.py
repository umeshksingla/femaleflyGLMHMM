####################################

# Usage: python regen_figures.py

####################################
import glob
import joblib
import os

from utilities import utils


if __name__  == '__main__':

    # Final figures use figures mostly from '20250721_185327_mushy' but some also from 20250711_032553_cheesecake and 20250714_152531_admin 20250723_030315_cenotaph
    # 20250804_211804_ride for auxem figures
    model_pkl_path = f'models/general_wt_fred_lr_temp/glm-hmm_5_cv/20250804_232412_consistency'

    # utils.enhance(model_pkl_path)
    # utils.generate_figures(model_pkl_path, savefig=1, display=0, override_fig_dir=False)
    utils.enhance_auxem(model_pkl_path, savefig=1, display=0)
    utils.generate_auxem_plots(model_pkl_path, savefig=1, display=0)
    # utils.generate_state_traces(model_pkl_path, dataset='wt_fred', savefig=1, display=0)
    # utils.generate_state_clips(model_pkl_path, savefig=1, display=0, gen_corr_video=True)
    # utils.generate_TAs(model_pkl_path, savefig=1, display=0)
    # utils.generate_trajs(model_pkl_path, savefig=1, display=0, gen_corr_video=False)

    # utils.generate_videos(model_pkl_path, override_vid_dir=True)

    # model_dir = 'models/general_cop/lrhmmci2_4/20250508_185109_fragrance'

    # for _ in glob.glob(model_dir + '/session*'):
    #     utils.generate_figures_single(_)

    # utils.generate_figures_all_singles_merged(model_dir, savefig=True, display=True, override_fig_dir=False)

