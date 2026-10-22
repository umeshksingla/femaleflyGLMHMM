import jax
import numpy as np
import jax.random as jr
import jax.numpy as jnp
from hmms.LRHMMFemaleFly import LRHMMFemaleFly
from hmms.LRFemaleFly import LRFemaleFly
from hmms.DiagGHMMFemaleFly import DiagGHMMFemaleFly
from hmms.ChanceFemaleFly import ChanceFemaleFly
from utilities import fitting, io
from utilities.io import ENABLE_FLOAT64

jax.config.update("jax_enable_x64", ENABLE_FLOAT64)


class LRHMMCustomInitFemaleFly(LRHMMFemaleFly):

    prefix = 'lrhmmci_'

    # def fit(self, batched_emissions, batched_inputs, batched_output_mn_std, do_reindex_params=False):
    #     print(f'Begin fitting chance...')
    #     chance = ChanceFemaleFly(self.data_config, self.model_config)
    #     chance.fit(batched_emissions, batched_inputs)
    #     chance_params = chance.learned_params
    #     self.chance_mu = chance_params['mu']
    #     self.chance_cov = chance_params['cov']
    #     print('chance fit.')
    #     print(f'Begin fitting {self.__class__.__name__}...')
    #     key = jr.PRNGKey(self.seed)

    #     # since the sessions are variable length, chunk the sessions.
    #     emissions_to_fit = io.chunk_data(batched_emissions, chunk_size=5000)
    #     inputs_to_fit = io.chunk_data(batched_inputs, chunk_size=5000)

    #     lr = LRFemaleFly(self.data_config, self.model_config)
    #     lr.fit(emissions_to_fit, inputs_to_fit)
    #     W = np.repeat(lr.learned_params['w'], repeats=self.num_states, axis=0)
    #     b = np.repeat(lr.learned_params['b'], repeats=self.num_states, axis=0)
    #     W = W + np.random.random(W.shape) * 1e-4
    #     b = b + np.random.random(b.shape) * 1e-4
    #     # print("W", W)
    #     print("LR global W and b computed.", W.shape, b.shape)
    #     em_params, em_lps = fitting.fitEMCustomInit(key, self.model, emissions_to_fit, train_inputs=inputs_to_fit,
    #                                                 emission_weights=W, emission_biases=b)
    #     self.learned_params = em_params
    #     if do_reindex_params:
    #         self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
    #     self.learned_lps = em_lps
    #     self.update_status()
    #     print(f'End fitting {self.__class__.__name__}...')
    #     return

    def fit(self, batched_emissions, batched_inputs, batched_output_mn_std, do_reindex_params=True):
        print(f'Begin fitting chance...')
        chance = ChanceFemaleFly(self.data_config, self.model_config)
        chance.fit(batched_emissions, batched_inputs)
        chance_params = chance.learned_params
        self.chance_mu = chance_params['mu']
        self.chance_cov = chance_params['cov']
        print('chance fit.')
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)

        # since the sessions are variable length, chunk the sessions.
        emissions_to_fit = io.chunk_data(batched_emissions, chunk_size=5000)
        inputs_to_fit = io.chunk_data(batched_inputs, chunk_size=5000)

        dghmm = DiagGHMMFemaleFly(self.data_config, self.model_config)
        dghmm.fit(batched_emissions, batched_inputs, batched_output_mn_std, do_reindex_params=False)

        p = dghmm.learned_params
        print("DGHMM params computed.")
        K, O = p.emissions.means.shape
        D = inputs_to_fit.shape[-1]
        em_params, em_lps = fitting.fitEMCustomInit(key, self.model, emissions_to_fit, train_inputs=inputs_to_fit,
                                                    initial_probs=p.initial.probs,
                                                    transition_matrix=p.transitions.transition_matrix,
                                                    emission_weights=jnp.zeros((K, O, D)),
                                                    emission_biases=p.emissions.means,
                                                    emission_covariances=jax.vmap(jnp.diag)(p.emissions.scale_diags ** 2)
                                                    )
        self.learned_params = em_params
        print(f'LR-HMM training loss: start {float(em_lps[0]):.4f}, end {float(em_lps[-1]):.4f}')
        if do_reindex_params:
            self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return
