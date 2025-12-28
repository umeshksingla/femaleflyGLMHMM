import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from hmms.InputDrivenLRHMMFemaleFly import InputDrivenLRHMMFemaleFly
from hmms.LRFemaleFly1 import LRFemaleFly1

from utilities import fitting

jax.config.update("jax_enable_x64", True)


def chunk_data(data_list, chunk_size):
    # chop long sequences into multiple shorter "chunks" of fixed length.
    chunked_data = []
    for seq in data_list:
        # Calculate how many full chunks we can make
        n_chunks = len(seq) // chunk_size
        if n_chunks > 0:
            # Keep only the part that fits perfectly into chunks
            cutoff = n_chunks * chunk_size
            # Reshape: (N_chunks, chunk_size, Features)
            reshaped = seq[:cutoff].reshape(n_chunks, chunk_size, -1)
            chunked_data.append(reshaped)
        else:
            print("Skipped.")

    # Stack all chunks from all sequences into one massive batch
    chunked_data = jnp.concatenate(chunked_data, axis=0)
    print("Data shapes (orig, chunked): ", len(data_list), chunked_data.shape)
    return chunked_data


class InputDrivenLRHMMCustomInitFemaleFly(InputDrivenLRHMMFemaleFly):

    prefix = 'id-glm-hmm'

    def fit(self, batched_emissions, batched_inputs, batched_output_mn_std):
        """
            batched_emissions: session by session
            batched_inputs:
            batched_output_mn_std:
        """
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)

        # since the sessions are variable length, chunk the sessions.
        emissions_to_fit = chunk_data(batched_emissions, chunk_size=5000)
        inputs_to_fit = chunk_data(batched_inputs, chunk_size=5000)

        lr = LRFemaleFly1(self.data_config, self.model_config)
        lr.fit(emissions_to_fit, inputs_to_fit)
        W = np.repeat(lr.learned_params['w'], repeats=self.num_states, axis=0)
        b = np.repeat(lr.learned_params['b'], repeats=self.num_states, axis=0)
        W = W + np.random.random(W.shape) * 1e-4
        b = b + np.random.random(b.shape) * 1e-4
        # print("W", W)
        print("LR global W and b computed.", W.shape, b.shape)
        em_params, em_lps = fitting.fitEMInputDrivenCustomInit(key, self.model, emissions_to_fit, train_inputs=inputs_to_fit,
                                                    emission_weights=W, emission_biases=b)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return
