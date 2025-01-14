import sys

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import joblib


# Load data
# data = joblib.load('simulated_data.pkl')
data = joblib.load('data/fly_data_False.pkl')
data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']

# Split into train and test
num_batches = data_config['num_batches']
num_train_batches = int(num_batches * 0.8)
print("num_train_batches", num_train_batches, num_batches)
train_emissions, train_inputs = emissions[:num_train_batches], inputs[:num_train_batches]


def exponential(X, w):
    return np.exp(X @ w)


def softplus(X, w):
    return np.log1p(np.exp(X @ w))


def compute_nll_poisson_glm_exp(w, X, Y, dt):
    z = exponential(X, w)
    L = np.dot(Y, np.log(z)) - dt*np.sum(z)
    return -L


def compute_nll_poisson_glm_softplus(w, X, Y, dt):
    z = softplus(X, w)
    L = np.dot(Y, np.log(z)) - dt*np.sum(z)
    return -L


btch_idx = 0
X = train_inputs[btch_idx]
fFS = np.abs(train_emissions[btch_idx, :, 0])   # only ffv for now
plt.hist(fFS, bins=100)
plt.show()
plt.close()

diff = np.abs(np.diff(fFS, prepend=0))
diff_p = np.percentile(diff[diff > 0], q=95)
Y = np.where(diff > diff_p, 1, 0)
print(np.unique(Y, return_counts=True))
plt.plot(Y[:500])
plt.plot(fFS[:500])
plt.show()
plt.close()

dtStim = 1
d = X.shape[-1]
X_aug = np.hstack([X, np.ones((X.shape[0], 1))])

# sys.exit()

w0 = np.random.rand(X_aug.shape[1])
result = optimize.minimize(compute_nll_poisson_glm_softplus, w0, args=(X_aug, Y, dtStim))
print(result)

w_MLE_poisson_exp_b = result.x
w_MLE_poisson_exp = w_MLE_poisson_exp_b[:-1]
w_MLE_poisson_exp = w_MLE_poisson_exp / np.linalg.norm(w_MLE_poisson_exp)

plt.figure(figsize=(7, 4))
t = np.arange(-d, 0)
# plt.plot(t, w_MLE_linearreg, '-o', label='linear-gaussian filter')
# plt.plot(t, STA_neuron1, '-o', label='neuron1 STA')
plt.plot(t, w_MLE_poisson_exp, '-o', label='poisson GLM w exp filter')
plt.ylabel('filter amplitude')
plt.xlabel('time (bins)')
plt.legend()
plt.show()
