####################################

# Usage: python compare_shuffled.py

# load a chance model ran on shuffled input data
# load a fit glmhmm model  on regular data
# get its params
# get the data logprob with shuffled inputs using the fit params

####################################
import glob
import numpy as np

from utilities import utils


if __name__  == '__main__':

    chance_shuffled_model_pkl_path = f'models/general_wt_shuffled/chance_0_cv/20260102_132745_sherbet'    # for shuffled data
    chance_model_pkl_path = f'models/general_wt/chance_0_cv/20260102_133220_doorway'    # for normal data
    idglmhmm_model_pkl_path = f'models/general_wt/20251229_041412_rice'    # for idglmhmm model
    ghmm_model_pkl_path = f'models/general_wt/ghmm_5_cv/20260102_143703_fireman'    # for blind hmm model

    chance_shuffled_model_pkl, chance_shuffled_data_config_pkl, chance_shuffled_model_config = utils.load_specific_path(chance_shuffled_model_pkl_path)
    chance_model_pkl, chance_data_config_pkl, chance_model_config = utils.load_specific_path(chance_model_pkl_path)
    idglmhmm_model_pkl, idglmhmm_data_config_pkl, idglmhmm_model_config = utils.load_specific_path(idglmhmm_model_pkl_path)
    ghmm_model_pkl, ghmm_data_config_pkl, ghmm_model_config = utils.load_specific_path(ghmm_model_pkl_path)

    model = ghmm_model_pkl['model']
    factor_bits_per_sec = idglmhmm_data_config_pkl['effective_fps'] / np.log(2)
    print(f'factor_bits_per_sec: {factor_bits_per_sec}')

    train_emissions = chance_shuffled_model_pkl['train_data']['train_emissions']
    train_shuffled_inputs = chance_shuffled_model_pkl['train_data']['train_inputs']
    # train_shuffled_lp = model.get_data_logprob(train_emissions, train_shuffled_inputs)
    shuffled_soft_emission_predictions, _, _, _, _ = model.predict(train_emissions, train_shuffled_inputs)
    train_shuffled_score = model.score(train_emissions, shuffled_soft_emission_predictions)
    train_shuffled_pearson = model.pearson(train_emissions, shuffled_soft_emission_predictions)
    # print("train_shuffled_lp", train_shuffled_lp * factor_bits_per_sec)
    print("train_shuffled_score", train_shuffled_score, "train_shuffled_pearson", train_shuffled_pearson)

    train_emissions = chance_model_pkl['train_data']['train_emissions']
    train_inputs = chance_model_pkl['train_data']['train_inputs']
    # train_lp = model.get_data_logprob(train_emissions, train_inputs)
    soft_emission_predictions, _, _, _, _ = model.predict(train_emissions, train_inputs)
    train_score = model.score(train_emissions, soft_emission_predictions)
    train_pearson = model.pearson(train_emissions, soft_emission_predictions)
    # print("train_lp", train_lp * factor_bits_per_sec)
    print("train_score", train_score, "train_pearson", train_pearson)


