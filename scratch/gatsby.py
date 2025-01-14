import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from scipy.stats import moment
import time

# def fsqrt(xi, wi):
#     return jnp.dot(wi, xi) / jnp.sqrt(n)

# w = 3 + 3 * jr.normal(subkey, shape=(n, ))
# x = 4 + 2 * jr.normal(subkey, shape=(n,))
# w = 3 + jr.lognormal(key1, 0.5, shape=(n, n))
# x = jr.beta(key2, 2, 5, shape=(n,))

s = time.time()
print("Started:")

n = 1000

mu_x = 4
var_x = 2

mu_w = 3
var_w = 3


def yi(_, w_, x_):
    np.random.seed(_)
    y = w_@x_ / np.sqrt(n)
    return y


def get_ys(__):
    # print(f"Run: {__}")
    w = np.random.normal(mu_w, var_w, (n, n))
    x = np.random.normal(mu_x, var_x, (n, ))
    run_ys = [yi(_+__, w[_], x) for _ in range(n)]
    # my_y_var = (2**2 + 4**2) * (3**2 + 3**2) - (4**2)*(3**2)
    # print(np.mean(run_ys), np.var(run_ys), my_y_var)
    return run_ys


# print(run_ys, len(run_ys))
n_runs = n
quants = np.array([[np.mean(get_ys(_)), np.var(get_ys(_))] for _ in range(n_runs)])    # get many many mus and vars
print(quants.shape)

mus = quants[:, 0]
vars = quants[:, 1]

mean_mu = np.mean(mus)
var_mu = np.var(mus)

mean_var = np.mean(vars)
var_var = np.var(vars)

my_mean_mu = jnp.sqrt(n) * mu_x * mu_w
my_var_mu = (mu_w**2) * (var_x**2)
print(f"Results SE[mu]={mean_mu}, SV[mu]={var_mu}, E[mu]={my_mean_mu}, V[mu]={my_var_mu}")

my_mean_var = (var_x**2 + mu_x**2) * (var_w ** 2)
# my_var_var # mean of variance is independent of n.
# so variance of variance will just be a finite constant in n -> inf conditions.
print(f"Results SE[var]={mean_var}, SV[var]={var_var}, E[var]={my_mean_var}, V[var]=")

print("Done:", time.time()-s, "secs")
