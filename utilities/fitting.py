import time

import jax.random as jr
import jax.numpy as jnp
import numpy as np


def fitEMCustomInit(key, hmm, train_emissions, train_inputs,
                    initial_probs=None,
                    transition_matrix=None,
                    emission_weights=None,
                    emission_biases=None,
                    ):
    print("=== START emCustomInit ==========")
    s = time.time()
    _, key = jr.split(key)
    em_params, em_props = hmm.initialize(key,
                                         initial_probs=initial_probs,
                                         transition_matrix=transition_matrix,
                                         emission_weights=emission_weights,
                                         emission_biases=emission_biases,
                                         )
    em_params, em_losses = hmm.fit_em_lrhmm_custom(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=50, verbose=True)
    train_emissions_size = np.sum([e.size for e in train_emissions])
    em_lps = -em_losses / train_emissions_size
    print("em_lps:", em_lps)
    ll_diff = np.diff(em_lps)
    # print(ll_diff)
    converged = np.abs(ll_diff[-1]) < 1e-3
    if not converged:
        print("!!! !!! EM not converged !!! !!!")
    e = time.time()
    print("=== END emCustomInit ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_lps


def fitEMInputDrivenCustomInit(key, hmm, train_emissions, train_inputs,
                    initial_probs=None,
                    emission_weights=None,
                    emission_biases=None,
                    ):
    print("=== START emCustomInit ==========")
    s = time.time()
    _, key = jr.split(key)
    em_params, em_props = hmm.initialize(key,
                                         initial_probs=initial_probs,
                                         emission_weights=emission_weights,
                                         emission_biases=emission_biases,
                                         )
    em_params, em_losses = hmm.fit_em_original(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=50, verbose=True)
    train_emissions_size = np.sum([e.size for e in train_emissions])
    em_lps = -em_losses / train_emissions_size
    print("em_lps:", em_lps)
    ll_diff = np.diff(em_lps)
    # print(ll_diff)
    converged = np.abs(ll_diff[-1]) < 1e-3
    if not converged:
        print("!!! !!! EM not converged !!! !!!")
    e = time.time()
    print("=== END emCustomInit ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_lps


def fitEMLogRHMMCustomInit(key, hmm, train_emissions, train_inputs,
                    initial_probs=None,
                    transition_matrix=None,
                    emission_weights=None,
                    emission_biases=None,
                    ):
    print("=== START emCustomInit ==========")
    s = time.time()
    _, key = jr.split(key)
    em_params, em_props = hmm.initialize(key, initial_probs=initial_probs,
                                         transition_matrix=transition_matrix,
                                         emission_weights=emission_weights,
                                         emission_biases=emission_biases,
                                         )
    em_params, em_losses = hmm.fit_em_logrhmm_custom(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=50, verbose=True)
    train_emissions_size = np.sum([e.size for e in train_emissions])
    em_lps = -em_losses / train_emissions_size
    # print("em_lps:", em_lps)
    ll_diff = np.diff(em_lps)
    print(ll_diff)
    converged = np.abs(ll_diff[-1]) < 1e-3
    if not converged:
        print("!!! !!! EM not converged !!! !!!")
    e = time.time()
    print("=== END emCustomInit ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_lps


def fitEM(key, hmm, train_emissions, train_inputs):
    print("=== START em ==========")
    s = time.time()
    fbgd_key, key = jr.split(key)
    em_params, em_props = hmm.initialize(key)
    em_params, em_losses = hmm.fit_em_lrhmm_custom(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=200, verbose=True)
    train_emissions_size = np.sum([e.size for e in train_emissions])
    em_lps = -em_losses / train_emissions_size
    ll_diff = np.diff(em_lps)
    print(ll_diff)
    converged = np.abs(ll_diff[-1]) < 1e-3
    if not converged:
        print("!!! !!! EM not converged !!! !!!")
    e = time.time()
    print("=== END em ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_lps

