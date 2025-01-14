import os
# import tensorflow_probability as tfp

from tensorflow_probability.substrates.jax import distributions as tfd

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tfd = tfp.distributions

# covariance_matrix = [[0, 0],
#                      [0, 0]]
# covariance_matrix = [[1, 0],
#                      [0, 1]]
covariance_matrix = [[-2.03379927e-03,  4.71475915e-03],
                     [ 4.71475915e-03,  4.10844721e-02]]
# covariance_matrix = [[ 0.07140106,  0.04058167],
#                      [ 0.04058167,  0.0682928 ]]
# covariance_matrix = [[ 1.85104686e+00, -6.26295277e-01],
#                      [-6.26295277e-01,  9.54241323e+00]]
p = tfd.MultivariateNormalFullCovariance(loc=[0,0], covariance_matrix=covariance_matrix)
q = tfd.Normal(loc=0, scale=-1)

print(p.log_prob([0,0]))
print(q.log_prob(0))


