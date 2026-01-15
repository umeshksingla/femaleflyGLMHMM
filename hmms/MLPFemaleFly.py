"""
This file is kept for reference only; it was not used in the paper.
"""

import os

import joblib

from sklearn.neural_network import MLPRegressor

# Load data
# data = joblib.load('simulated_data.pkl')
data = joblib.load('data/fly_data_cos.pkl')
data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']

# Split into train and test
num_batches = data_config['num_batches']
num_train_batches = int(num_batches * 0.8)
print("num_train_batches", num_train_batches, num_batches)
train_emissions, train_inputs = emissions[:num_train_batches], inputs[:num_train_batches]
test_emissions, test_inputs = emissions[num_train_batches:], inputs[num_train_batches:]

print("train and test shapes", train_emissions.shape, train_inputs.shape, test_emissions.shape, test_inputs.shape)

model_config = {}

# Fit train data
X_tr = train_inputs.reshape(-1, train_inputs.shape[-1])
Y_tr = train_emissions.reshape(-1, train_emissions.shape[-1])
print("X_tr.shape, Y_tr.shape", X_tr.shape, Y_tr.shape)

mlp_reg = MLPRegressor(random_state=1, alpha=0, early_stopping=True, validation_fraction=0.3, max_iter=500, verbose=True)
mlp_reg.fit(X_tr, Y_tr)

dump_filename = 'data/mlp_reg/mlp_reg_model_ckp.pkl'
os.makedirs(os.path.dirname(dump_filename), exist_ok=True)

model_ckp = {
        'train_data': {
            'train_emissions': train_emissions,
            'train_inputs': train_inputs,
        },
        'test_data': {
            'test_emissions': test_emissions,
            'test_inputs': test_inputs,
        },
        'data_config': data_config,
        'mlp_reg': mlp_reg,
    }

joblib.dump(model_ckp, dump_filename)

