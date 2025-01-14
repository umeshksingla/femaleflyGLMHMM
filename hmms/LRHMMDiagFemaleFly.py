import jax
from dynamax.hidden_markov_model import LinearRegressionHMM
from dynamax.hidden_markov_model.models.gaussian_hmm import DiagonalGaussianHMMEmissions
from LRHMMFemaleFly import LRHMMFemaleFly

# print("jax.config", jax.config.values)
jax.config.update("jax_enable_x64", True)


class LRHMMDiagFemaleFly(LRHMMFemaleFly):

    prefix = 'lrdiaghmm'

    def __init__(self, data_config, model_config):
        super().__init__(data_config, model_config)
        self.model = LinearRegressionHMM(num_states=self.model_config['num_states'],
                                    input_dim=self.data_config['input_dim'],
                                    emission_dim=self.data_config['emission_dim'],
                                    transition_matrix_stickiness=self.model_config['transition_matrix_stickiness'])
        self.model.emission_component = DiagonalGaussianHMMEmissions(self.model_config['num_states'], self.data_config['emission_dim'])
