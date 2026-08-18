"""
Linear Autoregressive Hidden Markov Model (ARHMM) for dynamax.
"""
from typing import NamedTuple, Optional, Tuple, Union
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Int, Float, Array
from jax.scipy.linalg import block_diag

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.linreg_hmm import LinearRegressionHMMEmissions, ParamsLinearRegressionHMMEmissions
# from dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMMEmissions, ParamsLinearRegressionHMMEmissions
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar
from dynamax.utils.utils import pytree_sum
from dynamax.utils.bijectors import RealToPSDBijector
from tensorflow_probability.substrates import jax as tfp
import optax

from library.inputdriven_transitions import InputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions


tfd = tfp.distributions
tfb = tfp.bijectors


class ParamsInputDrivenLinearAutoregressiveHMMEmissions(NamedTuple):
    weights: Union[Float[Array, "state_dim emission_dim total_input_dim"], ParameterProperties]
    biases: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    covs: Union[Float[Array, "state_dim emission_dim emission_dim"], ParameterProperties]


class ParamsInputDrivenLinearAutoregressiveHMM(NamedTuple):
    """Model parameters for a Linear Autoregressive HMM."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsInputDrivenHMMTransitions
    emissions: ParamsInputDrivenLinearAutoregressiveHMMEmissions


class InputDrivenLinearAutoregressiveHMMEmissions(LinearRegressionHMMEmissions):
    r"""Emissions for an Input Driven Linear Autoregressive HMM."""

    def __init__(self,
                 num_states: int,
                 total_input_dim: int,
                 emission_dim: int,
                 n_bases_per_feature,
                 num_lags: int = 1,
                 input_mask_by_emission=None):
        self.num_lags = num_lags
        self.emission_dim = emission_dim
        self.total_input_dim = total_input_dim
        self.n_bases_per_feature = n_bases_per_feature
        if input_mask_by_emission is None:
            input_mask_by_emission = jnp.ones((self.emission_dim, self.total_input_dim))
        self.input_mask_by_emission_full = jnp.broadcast_to(input_mask_by_emission, (num_states, emission_dim, total_input_dim))
        super().__init__(num_states, total_input_dim, emission_dim)

    def initialize(self,
                   key: Array = jr.PRNGKey(0),
                   method: str = "prior",
                   emission_weights: Optional[Float[Array, "num_states emission_dim total_input_dim"]] = None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]] = None,
                   emission_covariances: Optional[Float[Array, "num_states emission_dim emission_dim"]] = None,
                   emissions: Optional[Float[Array, "num_timesteps emission_dim"]] = None
                   ) -> Tuple[ParamsLinearRegressionHMMEmissions, ParamsLinearRegressionHMMEmissions]:

        if method.lower() == "prior":
            # technically there's an MNIW prior, but that's a bit complicated...
            key1, key2, key = jr.split(key, 3)
            _emission_weights = jnp.zeros((self.num_states, self.emission_dim, self.total_input_dim))
            _emission_weights = _emission_weights.at[:, :, :self.emission_dim].set(0.95 * jnp.eye(self.emission_dim))
            _emission_weights += 0.01 * jr.normal(key1, (self.num_states, self.emission_dim, self.total_input_dim))
            _emission_biases = jr.normal(key2, (self.num_states, self.emission_dim))
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))
        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsLinearRegressionHMMEmissions(
            weights=default(emission_weights, _emission_weights),
            biases=default(emission_biases, _emission_biases),
            covs=default(emission_covariances, _emission_covs))
        props = ParamsLinearRegressionHMMEmissions(
            weights=ParameterProperties(),
            biases=ParameterProperties(),
            covs=ParameterProperties(constrainer=RealToPSDBijector()))
        return params, props

    def m_step(self, params, props, batch_stats, m_step_state):
        def _single_m_step(stats):
            sum_w = stats['sum_w']
            sum_x = stats['sum_x']
            sum_y = stats['sum_y']
            sum_xxT = stats['sum_xxT']
            sum_xyT = stats['sum_xyT']
            sum_yyT = stats['sum_yyT']

            # L2 Regularization
            lambdaL2 = 1e6  # L2 penalty strength
            D = sum_xxT.shape[1]  # number of features
            I = jnp.eye(D + 1)  # (D+1)x(D+1), includes bias term
            I = I.at[-1, -1].set(0.0)   # set l2 for bias term to 0

            # Or, Laplacian Smoothing Regularization
            # Not needed in sparse basis transformed space but if you have too many cosines, then needed.
            lambda_laplace = 1e6  # Tune this!
            B = self.n_bases_per_feature
            AR = self.num_lags * self.emission_dim
            D_ = D - AR
            num_features = D_ // B

            # Create the Laplacian for a SINGLE feature series (size B x B)
            diag_main = jnp.full(B, 2.0).at[0].set(1.0).at[-1].set(1.0)     # Main diagonal: [1, 2, 2, ..., 2, 1]
            diag_off = jnp.full(B - 1, -1.0)                                        # Off-diagonals: [-1, -1, ..., -1]
            L_single = jnp.diag(diag_main) + jnp.diag(diag_off, k=1) + jnp.diag(diag_off, k=-1)

            # Tile it into a block diagonal matrix (size D x D)
            L = jnp.kron(jnp.eye(num_features), L_single)

            # autoregressive coefficients
            ar_diag_main = jnp.full(AR, 2.0).at[0].set(1.0).at[-1].set(1.0)
            ar_diag_off = jnp.full(AR - 1, -1.0)
            L_ar = jnp.diag(ar_diag_main) + jnp.diag(ar_diag_off, k=1) + jnp.diag(ar_diag_off, k=-1)

            # Join external inputs and autoregressive inputs laplacian reg matrices
            L = block_diag(L, L_ar)

            # Pad with 0s for the bias term so it becomes (D+1)x(D+1). We don't want to smooth the bias term.
            L = jnp.pad(L, ((0, 1), (0, 1)), mode='constant')

            # Make block matrices for stacking features (x) and bias (1)
            sum_x1x1T = jnp.block(
                [[sum_xxT,                   jnp.expand_dims(sum_x, 1)],
                 [jnp.expand_dims(sum_x, 0), jnp.expand_dims(sum_w, (0, 1))]]
            )
            sum_x1yT = jnp.vstack([sum_xyT, sum_y])

            # Solve for the optimal A, b, and Sigma
            penalty_matrix = lambdaL2 * I + lambda_laplace * L
            Ab = jnp.linalg.solve(sum_x1x1T + penalty_matrix, sum_x1yT).T
            Sigma = 1 / sum_w * (sum_yyT - Ab @ sum_x1yT)
            Sigma = 0.5 * (Sigma + Sigma.T)                 # for numerical stability
            Sigma = jnp.diag(jnp.diag(Sigma))               # diagonal
            Sigma = Sigma + jnp.eye(Sigma.shape[0]) * 1e-6
            return Ab[:, :-1], Ab[:, -1], Sigma

        emission_stats = pytree_sum(batch_stats, axis=0)
        As, bs, Sigmas = vmap(_single_m_step)(emission_stats)
        As_masked = As * self.input_mask_by_emission_full      # Zero out the irrelevant weights for an emission
        params = params._replace(weights=As_masked, biases=bs, covs=Sigmas)
        return params, m_step_state


class InputDrivenLinearAutoregressiveHMM(HMM):
    def __init__(self,
                 num_states: int,
                 external_input_dim: int,
                 emission_dim: int,
                 n_bases_per_feature,
                 num_lags: int=1,
                 input_mask_by_emission: Float[Array, "emission_dim input_dim"] = None,
                 input_mask_first: Float[Array, "input_dim"] = None,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]] = 1.1,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50):
        self.emission_dim = emission_dim
        self.external_input_dim = external_input_dim
        self.num_lags = num_lags
        self.total_input_dim = self.num_lags * self.emission_dim + self.external_input_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = InputDrivenHMMTransitions(num_states, self.total_input_dim, input_mask_first, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        emission_component = InputDrivenLinearAutoregressiveHMMEmissions(num_states, self.total_input_dim, emission_dim, n_bases_per_feature, num_lags, input_mask_by_emission)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's inputs.
        """
        return (self.total_input_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim emission_dim_times_num_lags"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsInputDrivenLinearAutoregressiveHMM(**params), ParamsInputDrivenLinearAutoregressiveHMM(**props)

    def compute_lagged_inputs(self,
                       emissions: Float[Array, "num_timesteps emission_dim"],
                       prev_emissions: Optional[Float[Array, "num_lags emission_dim"]]=None
    ) -> Float[Array, "num_timesteps {self.num_lags}*{self.emission_dim}"]:
        r"""Helper function to compute the matrix of lagged emissions.

        Args:
            emissions: $(T \times N)$ array of emissions
            prev_emissions: $(L \times N)$ array of previous emissions. Defaults to zeros.

        Returns:
            $(T \times N \cdot L)$ array of lagged emissions. These are the inputs to the fitting functions.
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        padded_emissions = jnp.vstack((prev_emissions, emissions))
        num_timesteps = len(emissions)
        return jnp.column_stack([padded_emissions[lag:lag+num_timesteps]
                                 for lag in reversed(range(self.num_lags))])
