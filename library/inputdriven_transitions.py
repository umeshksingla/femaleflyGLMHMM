"""
Input-driven State transitions.

Based on code from Dynamax (MIT License). Modified by Umesh Singla.

"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, Int, PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
import optax

from dynamax.utils.optimize import run_gradient_descent
from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMMTransitions
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.types import Scalar


class ParamsInputDrivenHMMTransitions(NamedTuple):
    """Parameters for the transitions of an input-driven HMM."""
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_states num_states"], ParameterProperties]


class InputDrivenHMMTransitions(HMMTransitions):
    """
    HMM transitions for an input-driven HMM.
    The transition probabilities depend on external inputs/covariates:
        P(z_t | z_{t-1}, u_t) where u_t are inputs at time t

    For each previous state j, we use multinomial logistic regression:
        P(z_t = k | z_{t-1} = j, u_t) = softmax(W_j @ u_t + b_j)[k]
    """

    def __init__(
            self,
            num_states: int,
            input_dim: int,
            input_mask_first = None,
            m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
            m_step_num_iters: int = 50,
            l2_penalty: float = 1.0,
    ):
        """
        Args:
            num_states: Number of discrete states
            input_dim: Dimensionality of input vectors
        """
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim
        self.l2_penalty = l2_penalty
        if input_mask_first is None:
            input_mask_first = jnp.ones((self.input_dim,))
        self.input_mask_first_full = jnp.broadcast_to(input_mask_first,(num_states, num_states, input_dim))
        print("IDHMMTransitions input_mask_by_emission", input_mask_first.shape, "input_mask_by_emission_full", self.input_mask_first_full.shape)

    def distribution(
            self,
            params: ParamsInputDrivenHMMTransitions,
            state: Union[int, Int[Array, ""]],
            inputs: Float[Array, " input_dim"]) -> tfd.Distribution:
        """
        Return the distribution over the next state given the current state and input.

        Compute logits: W[state] @ inputs + b[state]
        weights[state] has shape (input_dim, num_states)
        inputs has shape (input_dim,)
        Result has shape (num_states,)
        """
        if inputs is None:
            raise ValueError("Inputs must be provided for input-driven transitions")

        logits = params.weights[state] @ inputs + params.biases[state]
        return tfd.Categorical(logits=logits)

    def initialize(
            self,
            key=jr.PRNGKey(0),
            method: str = "prior",
            transition_weights=None,
            transition_biases=None,
            **kwargs
    ) -> Tuple[ParamsInputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions]:
        if method == "prior":
            # Initialize with small random weights (near zero) so transitions start near uniform
            key_w, key_b = jr.split(key)
            _weights = jr.normal(key_w, (self.num_states, self.num_states, self.input_dim)) * 0.01
            _biases = jr.normal(key_b, (self.num_states, self.num_states)) * 0.01
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsInputDrivenHMMTransitions(
            weights=default(transition_weights, _weights),
            biases=default(transition_biases, _biases))
        props = ParamsInputDrivenHMMTransitions(
            weights=ParameterProperties(),
            biases=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsInputDrivenHMMTransitions) -> Scalar:
        """Return the log-prior probability of the transition parameters.

        Currently, there is no prior so this function returns 0.
        """
        return 0.0
        # return tfd.Normal(0, 0.01).log_prob(params.weights).sum()

    def collect_suff_stats(self,
                           params: ParameterSet,
                           posterior: HMMPosterior,
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> PyTree:
        """Collect sufficient statistics for updating the transition distribution parameters.

        Args:
            params: transition distribution parameters
            posterior: posterior distribution over latent states
            inputs: optional inputs

        Returns:
            PyTree of sufficient statistics for updating the transition distribution

        """
        return posterior.trans_probs, inputs

    def m_step(self,
               params: ParameterSet,
               props: PropertySet,
               batch_stats: PyTree,
               m_step_state: Any,
               scale: float=1.0
    ) -> Tuple[ParameterSet, Any]:
        unc_params = to_unconstrained(params, props)

        # Minimize the negative expected log joint probability
        def neg_expected_log_joint(unc_params):
            """Compute the negative expected log joint probability."""
            params = from_unconstrained(unc_params, props)
            params = params._replace(weights=params.weights * self.input_mask_first_full)   # Zero out the irrelevant weights (Step 1)

            def _single_expected_log_like(stats):
                """Compute the expected log likelihood for a single sequence."""
                expected_transitions, inputs = stats
                log_trans_matrix = jnp.log(self._compute_transition_matrices(params, inputs))
                lp = jnp.sum(expected_transitions * log_trans_matrix)
                return lp

            log_prior = self.log_prior(params)
            batch_ells = vmap(_single_expected_log_like)(batch_stats)
            expected_log_joint = log_prior + batch_ells.sum()

            l2_reg = self.l2_penalty * jnp.sum(params.weights ** 2)
            # l1_reg = jnp.sum(jnp.abs(params.weights))
            total_penalty = l2_reg # + l1_reg
            return -expected_log_joint / scale + total_penalty

        # Run gradient descent
        unc_params, m_step_state, losses = \
            run_gradient_descent(neg_expected_log_joint,
                                 unc_params,
                                 self.m_step_optimizer,
                                 optimizer_state=m_step_state,
                                 num_mstep_iters=self.m_step_num_iters)

        # Return the updated parameters and optimizer state
        params = from_unconstrained(unc_params, props)
        params = params._replace(weights=params.weights * self.input_mask_first_full)   # Zero out the irrelevant weights (Step 2)
        return params, m_step_state

    def __getstate__(self):
        # Get parent's state first
        state = super().__getstate__() if hasattr(super(), '__getstate__') else self.__dict__.copy()
        # Remove the optimizer (it's stored in parent but we remove it here)
        state.pop('m_step_optimizer', None)
        return state

    def __setstate__(self, state):
        # Restore state
        if hasattr(super(), '__setstate__'):
            super().__setstate__(state)
        else:
            self.__dict__.update(state)
        # Recreate optimizer
        self.m_step_optimizer = optax.adam(1e-2)
