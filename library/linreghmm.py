"""
Based on code from Dynamax (MIT License). Modified by Umesh Singla.
"""
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import jax
from jaxtyping import Float, Array
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.hidden_markov_model.models.linreg_hmm import ParamsLinearRegressionHMM, ParamsLinearRegressionHMMEmissions
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar
from dynamax.utils.utils import pytree_sum
from dynamax.utils.bijectors import RealToPSDBijector
from tensorflow_probability.substrates import jax as tfp
from typing import NamedTuple, Optional, Tuple, Union

tfd = tfp.distributions
tfb = tfp.bijectors


class LinearRegressionHMMEmissionsCustom(HMMEmissions):
    def __init__(self,
                 num_states,
                 input_dim,
                 emission_dim,
                 n_bases_per_feature,
                 emission_l2_penalty,
                 input_mask_by_emission=None):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_matrices (_type_): _description_
            emission_biases (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        self.num_states = num_states
        self.input_dim = input_dim
        self.emission_dim = emission_dim
        self.emission_l2_penalty = emission_l2_penalty
        self.n_bases_per_feature = n_bases_per_feature
        if input_mask_by_emission is None:
            input_mask_by_emission = jnp.ones((self.emission_dim, self.input_dim))
        self.input_mask_by_emission_full = jnp.broadcast_to(input_mask_by_emission, (num_states, emission_dim, input_dim))
        # print("LRHMMEmissions input_mask_by_emission", input_mask_by_emission.shape, "input_mask_by_emission_full", self.input_mask_by_emission_full.shape)

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def initialize(self,
                   key=jr.PRNGKey(0),
                   method="prior",
                   emission_weights=None,
                   emission_biases=None,
                   emission_covariances=None,
                   emissions=None):
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            key, subkey = jr.split(key)  # Create a random seed for SKLearn.
            sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
            km = KMeans(self.num_states, random_state=int(sklearn_key)).fit(emissions.reshape(-1, self.emission_dim))
            _emission_weights = jnp.zeros((self.num_states, self.emission_dim, self.input_dim))
            _emission_biases = jnp.array(km.cluster_centers_)
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim)[None, :, :], (self.num_states, 1, 1))

        elif method.lower() == "prior":
            # TODO: Use an MNIW prior
            key1, key2, key = jr.split(key, 3)
            _emission_weights = 0.01 * jr.normal(key1, (self.num_states, self.emission_dim, self.input_dim))
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

    def distribution(self, params, state, inputs):
        prediction = params.weights[state] @ inputs
        prediction += params.biases[state]
        return tfd.MultivariateNormalFullCovariance(prediction, params.covs[state])

    def log_prior(self, params):
        return 0.0

    # Expectation-maximization (EM) code
    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        expected_states = posterior.smoothed_probs
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, inputs)
        sum_y = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xxT = jnp.einsum("tk,ti,tj->kij", expected_states, inputs, inputs)
        sum_xyT = jnp.einsum("tk,ti,tj->kij", expected_states, inputs, emissions)
        sum_yyT = jnp.einsum("tk,ti,tj->kij", expected_states, emissions, emissions)
        return dict(sum_w=sum_w, sum_x=sum_x, sum_y=sum_y, sum_xxT=sum_xxT, sum_xyT=sum_xyT, sum_yyT=sum_yyT)

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        def _single_m_step(stats):
            sum_w = stats['sum_w']
            sum_x = stats['sum_x']
            sum_y = stats['sum_y']
            sum_xxT = stats['sum_xxT']
            sum_xyT = stats['sum_xyT']
            sum_yyT = stats['sum_yyT']

            O = sum_yyT.shape[0]

            # L2 Regularization
            l2 = self.emission_l2_penalty   # L2 penalty strength
            D = sum_xxT.shape[1]            # number of features
            I = jnp.eye(D + 1)              # (D+1)x(D+1), includes bias term
            I = I.at[-1, -1].set(0.0)       # set L2 for bias term to 0

            # Laplacian Smoothing Regularization
            # Create the Laplacian for a SINGLE feature series (size B x B)
            # Not needed in sparse basis transformed space but if you have too many cosines, then needed.
            # Also, not needed for orthoginalized-basis
            B = self.n_bases_per_feature
            lambda_laplace = 0
            num_features = D // B
            diag_main = jnp.full(B, 2.0).at[0].set(1.0).at[-1].set(1.0)             # Main diagonal: [1, 2, 2, ..., 2, 1]
            diag_off = jnp.full(B - 1, -1.0)                                        # Off-diagonals: [-1, -1, ..., -1]
            L_single = jnp.diag(diag_main) + jnp.diag(diag_off, k=1) + jnp.diag(diag_off, k=-1)
            L = jnp.kron(jnp.eye(num_features), L_single)                           # Tile it into a block diagonal matrix (size D x D)
            L = jnp.pad(L, ((0, 1), (0, 1)), mode='constant')                       # Pad with 0s for the bias term so it becomes (D+1)x(D+1). We don't want to smooth the bias term.
            print("lambda_laplace used", lambda_laplace)

            # Make block matrices for stacking features (x) and bias (1)
            sum_x1x1T = jnp.block(
                [[sum_xxT,                   jnp.expand_dims(sum_x, 1)],
                 [jnp.expand_dims(sum_x, 0), jnp.expand_dims(sum_w, (0, 1))]]
            )
            sum_x1yT = jnp.vstack([sum_xyT, sum_y])

            penalty_matrix = l2 * I + lambda_laplace * L

            # Fit each output using only its allowed inputs, bias always allowed
            m_all = jnp.concatenate([jnp.broadcast_to(self.input_mask_by_emission_full[0], (O, D)), 
                                     jnp.ones((O, 1))], axis=1).astype(sum_x1x1T.dtype)   # bias always allowed

            def solve_one(m, c):                                                    # Solve for the optimal A, b, and Sigma
                return jnp.linalg.solve((sum_x1x1T + penalty_matrix) * jnp.outer(m, m) + jnp.diag(1.0 - m), m * c)

            Ab = vmap(solve_one)(m_all, sum_x1yT.T)                                 # (O, D+1); masked weights exactly 0
            safe_sum_w = jnp.maximum(sum_w, 1e-8)
            Sigma = (sum_yyT - Ab @ sum_x1yT - sum_x1yT.T @ Ab.T + Ab @ sum_x1x1T @ Ab.T) / safe_sum_w
            Sigma = jnp.diag(jnp.maximum(jnp.diag(Sigma), 1e-3))                    # diagonal and floor the variances (so they stay above measurement noise)

            # Ab = jnp.linalg.solve(sum_x1x1T + penalty_matrix, sum_x1yT).T
            # jax.debug.print("sum_x1x1T={x} anynans={y} anynans={z}", x=sum_x1x1T, y=jnp.isnan(sum_x1x1T).any(), z=jnp.isnan(sum_x1yT).any())
            # Ab = jnp.linalg.solve(sum_x1x1T, sum_x1yT).T
            # jax.debug.print("sum_w={x}", x=sum_w)
            # safe_sum_w = jnp.maximum(sum_w, 1e-8)

            # Sigma = 1 / sum_w * (sum_yyT - Ab @ sum_x1yT)
            # jax.debug.print("Sigma={x}", x=Sigma)
            # jax.debug.print("sum_yyT={x} Absum_x1yT={y}", x=sum_yyT, y=Ab @ sum_x1yT)
            # jax.debug.print("sum_x1yT={x} Ab={y}", x=sum_x1yT, y=Ab)
            # Sigma = 0.5 * (Sigma + Sigma.T)                 # for numerical stability
            # Sigma = jnp.diag(jnp.diag(Sigma))   + 1e-3 * jnp.eye(Sigma.shape[0])      # diagonal
            return Ab[:, :-1], Ab[:, -1], Sigma

        emission_stats = pytree_sum(batch_stats, axis=0)
        As, bs, Sigmas = vmap(_single_m_step)(emission_stats)
        As_masked = As * self.input_mask_by_emission_full      # Zero out the irrelevant weights for an emission
        # print("masked A")
        params = params._replace(weights=As_masked, biases=bs, covs=Sigmas)
        return params, m_step_state


class LinearRegressionHMMCustom(HMM):
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int,
                 n_bases_per_feature: int,
                 emission_l2_penalty: float,
                 input_mask_by_emission: Float[Array, "emission_dim input_dim"]=None,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = LinearRegressionHMMEmissionsCustom(num_states, input_dim, emission_dim, n_bases_per_feature, emission_l2_penalty, input_mask_by_emission)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_weights: manually specified emission weights.
            emission_biases: manually specified emission biases.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsLinearRegressionHMM(**params), ParamsLinearRegressionHMM(**props)
