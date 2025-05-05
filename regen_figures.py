####################################

# Usage: python regen_figures.py

####################################
import glob

from utilities import utils


if __name__ == '__main__':

    model_pkl_path = f'models/general_cop/lrhmmci_4_cv/20250428_180407_mantle/'
    utils.generate_figures(model_pkl_path,
                           savefig=1, display=0, override_fig_dir=False)

    # for s in [3, 4]:
    #     model_pkl_paths = sorted(glob.glob(f'models/general_cop/lrhmmci_{s}_cv/**/'))
    #     for model_pkl_path in model_pkl_paths:
    #         utils.generate_figures(model_pkl_path,
    #                                savefig=1, display=0, override_fig_dir=False)
