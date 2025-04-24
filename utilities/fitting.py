import time

import jax.random as jr
import jax.numpy as jnp
import optax


def fitEMCustomInit(key, hmm, train_emissions, train_inputs,
                    initial_probs=None,
                    transition_matrix=None,
                    emission_weights=None,
                    emission_biases=None,
                    emission_covariances=None):
    print("=== START emCustomInit ==========")
    s = time.time()
    _, key = jr.split(key)
    em_params, em_props = hmm.initialize(key, initial_probs=initial_probs,
                                         transition_matrix=transition_matrix,
                                         emission_weights=emission_weights,
                                         emission_biases=emission_biases,
                                         emission_covariances=emission_covariances,
                                         )
    em_params, em_losses = hmm.fit_em(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=100, verbose=True)
    em_lps = -em_losses / train_emissions.size
    # print("em_lps:", em_lps)
    e = time.time()
    print("=== END emCustomInit ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_lps


def fitEM(key, hmm, train_emissions, train_inputs):
    print("=== START em ==========")
    s = time.time()
    fbgd_key, key = jr.split(key)
    em_params, em_props = hmm.initialize(key)
    em_params, em_losses = hmm.fit_em(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=200, verbose=True)
    em_lps = -em_losses / train_emissions.size
    e = time.time()
    print("=== END em ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_lps


def fitSGDCustomInit(key, hmm, train_emissions, train_inputs,
                    initial_probs=None,
                    transition_matrix=None,
                    emission_weights=None,
                    emission_biases=None,
                    emission_covariances=None):
    print("=== START sgdCustomInit ==========")
    s = time.time()
    fbgd_key, key = jr.split(key)
    fbgd_params, fbgd_props = hmm.initialize(key, initial_probs=initial_probs,
                                         transition_matrix=transition_matrix,
                                         emission_weights=emission_weights,
                                         emission_biases=emission_biases,
                                         emission_covariances=emission_covariances,
                                         )
    fbgd_params, fbgd_losses = hmm.fit_sgd(fbgd_params,
                                             fbgd_props,
                                             jnp.asarray(train_emissions),
                                             jnp.asarray(train_inputs),
                                             optimizer=optax.sgd(learning_rate=1e-2, momentum=0.95),
                                             batch_size=5,
                                             num_epochs=100,
                                             key=fbgd_key)
    # fbgd_lps = -fbgd_losses / train_emissions.size    # NO NEED?
    fbgd_lps = fbgd_losses
    e = time.time()
    print(f"Time taken to fit SGD: {round(e - s, 2)}s")
    print("=== END sgdCustomInit ==========")
    return fbgd_params, fbgd_lps


def fitSGD(key, hmm, train_emissions, train_inputs):
    print("=== START sgd ==========")
    s = time.time()
    fbgd_key, key = jr.split(key)
    fbgd_params, fbgd_props = hmm.initialize(key)
    fbgd_params, fbgd_losses = hmm.fit_sgd(fbgd_params,
                                             fbgd_props,
                                             train_emissions,
                                             train_inputs,
                                             optimizer=optax.sgd(learning_rate=1e-2, momentum=0.95),
                                             batch_size=5,
                                             num_epochs=400,
                                             key=fbgd_key)
    # fbgd_lps = -fbgd_losses / train_emissions.size    # NO NEED?
    fbgd_lps = fbgd_losses
    e = time.time()
    print(f"Time taken to fit SGD: {round(e - s, 2)}s")
    print("=== END sgd ==========")
    return fbgd_params, fbgd_lps



