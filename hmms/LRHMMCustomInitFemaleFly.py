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

    def fit(self, batched_emissions, batched_inputs, batched_output_mn_std, do_reindex_params=True, dghmm=None):
        print(f'Begin fitting {self.__class__.__name__}...')
        print(f'Begin fitting chance...')
        chance = ChanceFemaleFly(self.data_config, self.model_config)
        chance.fit(batched_emissions, batched_inputs)
        chance_params = chance.learned_params
        self.chance_mu = chance_params['mu']
        self.chance_cov = chance_params['cov']
        print('chance fit.')
        key = jr.PRNGKey(self.seed)

        # since the sessions are variable length, chunk the sessions.
        emissions_to_fit = io.chunk_data(batched_emissions, chunk_size=5000)
        inputs_to_fit = io.chunk_data(batched_inputs, chunk_size=5000)

        if dghmm is None:
            print("DGHMM not found.")
            dghmm = DiagGHMMFemaleFly(self.data_config, self.model_config)
            dghmm.fit(batched_emissions, batched_inputs, batched_output_mn_std, do_reindex_params=False)
            print("DGHMM params computed.")
        else:
            print("DGHMM found.")
        self.dghmm = dghmm
        p = dghmm.learned_params
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
        print(f'LR-HMM training lps: start {float(em_lps[0]):.4f}, end {float(em_lps[-1]):.4f}')
        if do_reindex_params:
            self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return
