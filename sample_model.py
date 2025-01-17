import joblib

from hmms.BaseFemaleFly import BaseFemaleFly
from hmms.LRHMMFemaleFly import LRHMMFemaleFly
from hmms.LRFemaleFly import LRFemaleFly
from hmms.GHMMFemaleFly import GHMMFemaleFly
from hmms.ChanceFemaleFly import ChanceFemaleFly

from utilities import utils

if __name__ == '__main__':

    MODEL = LRFemaleFly

    data = joblib.load(f'../data/fly_data_cos=4_ortho_o=15.pkl')
    model_config = {
        # 'num_states': 15,
        # 'transition_matrix_stickiness': 10,
    }

    data_config, emissions, inputs, output_indices = data['data_config'], data['emissions'], data['inputs'], data['output_indices']
    session_keys = data_config['session_keys']
    num_batches = data_config['num_sessions']
    num_train_batches = int(num_batches * 0.8)
    train_emissions, train_inputs, train_session_keys = emissions[:num_train_batches], inputs[:num_train_batches], session_keys[:num_train_batches]
    test_emissions, test_inputs, test_session_keys = emissions[num_train_batches:], inputs[num_train_batches:], session_keys[num_train_batches:]

    model = MODEL(data_config, model_config)
    model.fit(train_emissions, train_inputs)
    # test_emission_predictions, test_z_predictions = model.predict(test_emissions, test_inputs)

    dump_filepath = utils.getafilepath(f'{model.prefix}_{model.model_config["num_states"]}')
    print(">> Saving at:", dump_filepath)
    utils.save(model, train_emissions, train_inputs, train_session_keys, test_emissions, test_inputs, test_session_keys, output_indices, dump_filepath)

    print(model.score(train_emissions, train_inputs))

    print(model.score(test_emissions, test_inputs))
    print(model.score_by_z_and_o(test_emissions, test_inputs))
    print(model.correlation_by_o(test_emissions, test_inputs))
