import time

import jax.random as jr
import optax

from plotting.plots import print_params


def fitEM(key, hmm, train_emissions, train_inputs):
    print("=== START em ==========")
    s = time.time()
    fbgd_key, key = jr.split(key)
    em_params, em_props = hmm.initialize(key)
    em_params, em_losses = hmm.fit_em(em_params,
                                        em_props,
                                        train_emissions,
                                        train_inputs,
                                        num_iters=100, verbose=True)
    em_lps = -em_losses / train_emissions.size
    e = time.time()
    print("=== END em ==========")
    print(f"Time taken to fit EM: {round(e - s, 2)}s")
    return em_params, em_lps


# fbgd_params, fbgd_lps = fitting.fitSGD(key, lrhmm, train_emissions, train_inputs)
# learned_params = fbgd_params
# learned_lps = fbgd_lps
def fitSGD(key, lrhmm, train_emissions, train_inputs):
    print("=== START sgd ==========")
    s = time.time()
    fbgd_key, key = jr.split(key)
    fbgd_params, fbgd_props = lrhmm.initialize(key)
    fbgd_params, fbgd_losses = lrhmm.fit_sgd(fbgd_params,
                                             fbgd_props,
                                             train_emissions,
                                             train_inputs,
                                             optimizer=optax.sgd(learning_rate=1e-2, momentum=0.95),
                                             batch_size=1,
                                             num_epochs=400,
                                             key=fbgd_key)
    # fbgd_lps = -fbgd_losses / train_emissions.size    # NO NEED?
    fbgd_lps = fbgd_losses
    e = time.time()
    print(f"Time taken to fit SGD: {round(e - s, 2)}s")
    print("== FITTED (SGD) ===========")
    print_params(fbgd_params)
    print("=== END sgd ==========")
    return fbgd_params, fbgd_lps



