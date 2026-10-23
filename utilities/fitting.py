import time

import jax.random as jr
import jax.numpy as jnp
import numpy as np


def fitEMCustomInit(key, hmm, train_emissions, train_inputs,
                    initial_probs=None,
                    transition_matrix=None,
                    emission_weights=None,
                    emission_biases=None,
                    emission_covariances=None,
                    ):
    print("=== START emCustomInit ==========")
    s = time.time()
    _, key = jr.split(key)
    em_params, em_props = hmm.initialize(key,
                                         initial_probs=initial_probs,
                                         transition_matrix=transition_matrix,
                                         emission_weights=emission_weights,
                                         emission_biases=emission_biases,
                                         emission_covariances=emission_covariances,
                                         )
    em_params = em_params._replace(     # replace init probs and tr matrix as the .initialize does not allow changing those for some classes, like StandardHMMTransitions
        initial=em_params.initial._replace(probs=initial_probs),
        transitions=em_params.transitions._replace(transition_matrix=transition_matrix),
    )
    em_params, em_log_probs = hmm.fit_em_coverged(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=300, tol=1e-5, verbose=True)
    train_emissions_size = sum(e.shape[0] for e in train_emissions)
    em_log_probs_per_frame = em_log_probs / train_emissions_size
    ll_diff = np.diff(em_log_probs_per_frame)
    print("em_log_probs_per_frame::", em_log_probs_per_frame, "total iters:", len(em_log_probs))
    # print("ll_diff", ll_diff)
    converged = np.abs(ll_diff[-1]) < 1e-3
    if not converged:
        print("!!! !!! EM not converged !!! !!!")
    e = time.time()
    print("=== END emCustomInit ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_log_probs_per_frame


def fitEMInputDrivenCustomInit(key, hmm, train_emissions, train_inputs,
                    initial_weights=None,
                    initial_biases=None,
                    transition_weights=None,
                    transition_biases=None,
                    emission_weights=None,
                    emission_biases=None,
                    emission_covariances=None,
                    ):
    print("=== START emInputDrivenCustomInit ==========")
    s = time.time()
    _, key = jr.split(key)
    em_params, em_props = hmm.initialize(key,
                                         transition_weights=transition_weights,
                                         transition_biases=transition_biases,
                                         emission_weights=emission_weights,
                                         emission_biases=emission_biases,
                                         emission_covariances=emission_covariances,
                                         )
    em_params = em_params._replace(     # replace init weights and biases as the .initialize does not allow changing those for some classes, like InputDrivenHMMInitialState
        initial=em_params.initial._replace(weights=initial_weights, biases=initial_biases),
    )
    em_params, em_log_probs = hmm.fit_em_coverged(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=100, tol=1e-4, verbose=True)
    train_emissions_size = sum(e.shape[0] for e in train_emissions)
    em_log_probs_per_frame = em_log_probs / train_emissions_size
    ll_diff = np.diff(em_log_probs_per_frame)
    print("em_log_probs_per_frame::", em_log_probs_per_frame, "total iters:", len(em_log_probs))
    # print("ll_diff", ll_diff)
    converged = np.abs(ll_diff[-1]) < 1e-3
    if not converged:
        print("!!! !!! EM not converged !!! !!!")
    e = time.time()
    print("=== END emInputDrivenCustomInit ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_log_probs_per_frame


def fitDiagEM(hmm, em_params, em_props, train_emissions, train_inputs):
    print("=== START em ==========")
    s = time.time()
    em_params, em_log_probs = hmm.fit_em_coverged(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=300, tol=1e-5, verbose=True)
    train_emissions_size = sum(e.shape[0] for e in train_emissions)
    em_log_probs_per_frame = em_log_probs / train_emissions_size
    ll_diff = np.diff(em_log_probs_per_frame)
    print("em_log_probs_per_frame::", em_log_probs_per_frame, "total iters:", len(em_log_probs))
    converged = np.abs(ll_diff[-1]) < 1e-3
    if not converged:
        print("!!! !!! EM not converged !!! !!!")
    e = time.time()
    print("=== END em ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_log_probs_per_frame


## 10/23/26: Definition of ssm.fit_em to be replaced in dynamax.
# def fit_em_coverged(self,
#                     params,
#                     props,
#                     emissions,
#                     inputs=None,
#                     num_iters=50,
#                     verbose=True,
#                     tol=None,
#                     patience=5):
#     batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
#     batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)
#     n_frames = batch_emissions.shape[0] * batch_emissions.shape[1]

#     @jit
#     def em_step(params, m_step_state):
#         batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
#         lp = self.log_prior(params) + lls.sum()
#         params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
#         return params, m_step_state, lp

#     log_probs = []
#     best, best_params, since_best = -np.inf, params, 0
#     m_step_state = self.initialize_m_step_state(params, props)
#     pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
#     for _ in pbar:
#         new_params, m_step_state, marginal_logprob = em_step(params, m_step_state)
#         log_probs.append(marginal_logprob)
#         if tol is not None:
#             ll = float(marginal_logprob) / n_frames     # per frame, for params before this M-step
#             if not np.isfinite(ll):
#                 break
#             if ll > best + tol:
#                 best, best_params, since_best = ll, params, 0
#             else:
#                 since_best += 1
#                 if since_best >= patience:
#                     break
#         params = new_params
#     return (best_params if tol is not None else params), jnp.array(log_probs)
