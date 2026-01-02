####################################

# Usage: python regen_figures.py

####################################
import glob
from utilities import utils


if __name__  == '__main__':

    # model_pkl_path = f'models/jan1_kfoldcv_wt_female/id-glm-hmm_6_cv/20260101_235037_footage'    # for wt
    # model_pkl_path = f'models/final_wt_fred/20251229_040811_costume'    # for wt_fred

    # utils.enhance(model_pkl_path)
    # utils.enhance_auxem(model_pkl_path, savefig=1, display=0)

    # utils.generate_figures(model_pkl_path, savefig=1, display=0, override_fig_dir=False)
    # utils.generate_state_filters(model_pkl_path, savefig=1, display=0)
    # utils.generate_together_figures(model_pkl_path, savefig=1, display=0)
    # utils.generate_trajs(model_pkl_path, savefig=1, display=0, gen_corr_video=False)

    # utils.enhance_auxem(model_pkl_path, savefig=1, display=0)
    # utils.generate_auxem_plots(model_pkl_path, savefig=1, display=0)
    #
    # utils.generate_state_traces(model_pkl_path, dataset='wt', savefig=1, display=0)
    #
    # utils.generate_TAs(model_pkl_path, savefig=1, display=0)
    #
    # utils.generate_state_clips(model_pkl_path, savefig=1, display=0, gen_corr_video=True)
    # utils.generate_videos(model_pkl_path, override_vid_dir=True)

    model_dir = 'models/jan1_kfoldcv_wt_female/id-glm-hmm_5_cv/'

    for path in sorted(list(glob.glob(model_dir + '/20260102_*/'))):
        print("path", path)
        # if ('duel' in path) or ('duration' in path) or ('sink' in path) or ('communicant' in path):
        #     print("skipped.")
        #     continue
        utils.generate_figures(path, savefig=1, display=0, override_fig_dir=False)
        utils.generate_state_filters(path, savefig=1, display=0)

    # utils.generate_figures_all_singles_merged(model_dir, savefig=True, display=True, override_fig_dir=False)

