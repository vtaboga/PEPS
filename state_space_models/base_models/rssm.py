import jax
import jax.numpy as jnp
import haiku as hk
import gin

from tensorflow_probability.substrates import jax as tfp
from collections import namedtuple
from typing import NamedTuple, List, Dict

State = namedtuple('State', ['mean', 'std', 'stoch', 'deter'])
PriorPostState = namedtuple('PriorPostState', ['prior', 'posterior'])


def get_distribution(state):
    """use tensorflow probability for kl divergence"""
    return tfp.distributions.MultivariateNormalDiag(state.mean, state.std)


def get_features(state):
    res = jnp.concatenate([state.stoch, state.deter], 0)
    return res


class MLP(hk.Module):
    def __init__(self, output_dim: int, depth: int, hidden_dim: int, act: str = 'relu', last_layer_act: str = None):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.last_layer_activation = last_layer_act
        self.activation = act
        if act not in ['relu', 'tanh', 'elu'] and act is not None:
            raise ValueError(f'activation type {act} is not implemented yet')

    def __call__(self, x):
        for _ in range(self.depth):
            x = hk.Linear(self.hidden_dim)(x)
            if self.activation is not None:
                if self.activation == 'relu':
                    x = jax.nn.relu(x)
                elif self.activation == 'elu':
                    x = jax.nn.elu(x)
                else:
                    x = jax.nn.tanh(x)
        x = hk.Linear(self.output_dim)(x)
        if self.last_layer_activation is not None:
            if self.last_layer_activation == 'relu':
                x = jax.nn.relu(x)
            elif self.last_layer_activation == 'elu':
                x = jax.nn.elu(x)
            else:
                x = jax.nn.tanh(x)
        return x


class TransitionModel(hk.Module):
    def __init__(self, stoch, deter, hidden, activation, name=None):
        super().__init__(name=name)
        self.stoch = stoch
        self.deter = deter
        self.hidden = hidden
        self.activation = activation
        self.transition_model_1 = MLP(
            output_dim=hidden,
            depth=1,
            hidden_dim=hidden,
            act=activation,
            last_layer_act=activation
        )
        self.gru_cell = hk.GRU(deter)
        self.transition_model_2 = MLP(
            output_dim=2 * stoch,
            depth=1,
            hidden_dim=hidden,
            act=activation,
            last_layer_act=None
        )

    def __call__(self, state: NamedTuple, action: jnp.array, key: jnp.array) -> NamedTuple:
        """ prev_state: last observed state
            compute the next state prior distribution"""

        # Apply the model
        x = jnp.concatenate([state.stoch, action], -1)
        x = self.transition_model_1(x)
        x, new_deter_state = self.gru_cell(x, state.deter)
        x = self.transition_model_2(x)
        # compute new stochastic state
        mean_stoch_state, std_stoch_state = jnp.split(x, 2, -1)
        std_stoch_state = jax.nn.softplus(std_stoch_state) + 0.1
        new_stoch_state = tfp.distributions.MultivariateNormalDiag(mean_stoch_state, std_stoch_state).sample(
            seed=key
        )
        next_state_prior = State(
            mean=mean_stoch_state,
            std=std_stoch_state,
            stoch=new_stoch_state,
            deter=new_deter_state
        )

        return next_state_prior


def initial_state(deter, stoch):
    """respect the state order mean, std, stoch, deter"""

    state = State(
        mean=jnp.zeros(stoch),
        std=jnp.zeros(stoch),
        stoch=jnp.zeros(stoch),
        deter=jnp.zeros(deter)
    )
    return state


class ObservationModel(hk.Module):
    """Recurrent State Space Model.
        Batch dimensions should be handled prior to calling this class"""

    def __init__(self, stoch, deter, hidden, activation, name=None):
        super().__init__(name=name)
        self.stoch = stoch
        self.deter = deter
        self.hidden = hidden
        self.activation = activation

        self.obs_model = MLP(
            output_dim=2 * stoch,
            depth=2,
            hidden_dim=hidden,
            act=activation,
            last_layer_act=activation
        )

    def __call__(
        self,
        transition: hk.transform(TransitionModel),
        transition_params: Dict,
        prev_state: NamedTuple,
        prev_action: jnp.array,
        latent_observation: jnp.array,
        key: jnp.array,
        min_std: float = 0.1
    ) -> NamedTuple:
        """This method first compute the new stochastic state prior distribution using only the previous action and
        state. It then computes the posterior distribution of the stochastic state using the observation"""

        subkey1, subkey2 = jax.random.split(key)

        next_state_prior = transition.apply(transition_params, None, prev_state, prev_action, subkey1)
        x = jnp.concatenate([next_state_prior.deter, latent_observation], -1)
        x = self.obs_model(x)
        mean_posterior_state, std_posterior_state = jnp.split(x, 2, -1)
        std_posterior_state = jax.nn.softplus(std_posterior_state) + min_std
        new_stoch_posterior_state = tfp.distributions.MultivariateNormalDiag(
            mean_posterior_state,
            std_posterior_state
        ).sample(seed=subkey2)
        next_state_posterior = State(
            mean=mean_posterior_state,
            std=std_posterior_state,
            stoch=new_stoch_posterior_state,
            deter=next_state_prior.deter
        )

        state = PriorPostState(
            prior=next_state_prior,
            posterior=next_state_posterior
        )

        return state


class DenseDecoder(hk.Module):
    def __init__(self, layers, units, last_layer, activation='relu', name=None):
        super().__init__(name=name)
        self.layers = layers
        self.units = units
        self.last_layer = last_layer
        self.activation = activation
        self.decoder = MLP(output_dim=last_layer, depth=layers, hidden_dim=units, act=activation)

    def __call__(self, features):
        return self.decoder(features)


class DenseEncoder(hk.Module):
    def __init__(self, layers, units, last_layer, activation='relu', name=None):
        super().__init__(name=name)
        self.layers = layers
        self.units = units
        self.last_layer = last_layer
        self.activation = activation
        self.encoder = MLP(output_dim=last_layer, depth=layers, hidden_dim=units, act=activation)

    def __call__(self, features):
        return self.encoder(features)


class RecurrentStateSpacePredictionModel:

    def __init__(
        self,
        state_size,
        observed_state_size,
        encoder_depth,
        encoder_activation,
        decoder_activation,
        decoder_depth,
        stochastic_size,
        deterministic_size,
        latent_state_size,
        transition_model_activation,
        action_size,
    ):

        def apply_dense_encoder(x):
            encoder = DenseEncoder(
                layers=encoder_depth,
                units=latent_state_size,
                last_layer=latent_state_size,
                activation=encoder_activation
            )
            return encoder(x)

        def apply_dense_decoder(x):
            decoder = DenseDecoder(
                layers=decoder_depth,
                units=latent_state_size,
                last_layer=state_size,
                activation=decoder_activation
            )
            return decoder(x)

        def apply_transition_model(x, y, z):
            transition_model = TransitionModel(
                stoch=stochastic_size,
                deter=deterministic_size,
                hidden=latent_state_size,
                activation=transition_model_activation,
            )
            return transition_model(x, y, z)

        def apply_observation(transition, params, x, y, z, k):
            observe = ObservationModel(
                stoch=stochastic_size,
                deter=deterministic_size,
                hidden=latent_state_size,
                activation=transition_model_activation
            )
            return observe(transition, params, x, y, z, k)

        self.deter = deterministic_size
        self.stoch = stochastic_size
        self.latent_state_size = latent_state_size
        self.observed_state_size = observed_state_size
        self.action_size = action_size
        self.decoder = hk.transform(apply_dense_decoder)
        self.encoder = hk.transform(apply_dense_encoder)
        self.transition_model = hk.transform(apply_transition_model)
        self.observation_model = hk.transform(apply_observation)

    def __call__(self, params, timestamps, lags_timestamps, inputs, actions, key):
        """
        :param params:
        :param timestamps:
        :param lags_timestamps:
        :param inputs:
        :param actions:
        :param key:
        :param mode: wheter to take the mode or a sample of the trajectory
        :return: Trajectory of prior states (predictions)
        """

        subkey, key = jax.random.split(key)
        features = self.call_features(
            params=params,
            timestamps=timestamps,
            lags_timestamps=lags_timestamps,
            inputs=inputs,
            actions=actions,
            key=subkey
        )
        predicted_states = self.get_traj_mode(features)

        return predicted_states

    def sample_predictions(self, params, timestamps, lags_timestamps, inputs, actions, key):
        subkey, key = jax.random.split(key)
        features = self.call_features(
            params=params,
            timestamps=timestamps,
            lags_timestamps=lags_timestamps,
            inputs=inputs,
            actions=actions,
            key=subkey
        )
        sub_key, key = jax.random.split(key)
        predicted_states = self.sample_traj(features, key=sub_key)

        return predicted_states

    def call_features(self, params, timestamps, lags_timestamps, inputs, actions, key):

        # encode the state observations
        lags_observations, lags_actions = inputs[:, :self.observed_state_size], inputs[:, self.observed_state_size:]
        latent_encoder_inputs = jax.vmap(self.encoder.apply, in_axes=(None, None, 0))(
            params['encoder'],
            None,
            lags_observations
        )
        # compute the initial state (posterior at t=0)
        sub_key, key = jax.random.split(key)
        state, _ = self.encode_observations(
            params=params,
            lags_observations=latent_encoder_inputs,
            lags_actions=lags_actions,
            rng=sub_key
        )
        # get prior distributions (predictions)
        sub_key, key = jax.random.split(key)
        prior_states_trajectory = self.predict_trajectory(
            params,
            state,
            actions,
            sub_key
        )
        features = self.get_trajectory_features(prior_states_trajectory)
        features = self.decode_trajectory(params['decoder'], features)

        return features

    def call_training(self, params, timestamps, lags_timestamps, inputs, future_states, actions, key):
        """
        :param timestamps:
        :param lags_timestamps:
        :param inputs:
        :param future_states:
        :param actions:
        :param key:
        :return:
        """

        # encode the state observations
        lags_observations, lags_actions = inputs[:, :self.observed_state_size], inputs[:, self.observed_state_size:]
        latent_encoder_inputs = jax.vmap(self.encoder.apply, in_axes=(None, None, 0))(
            params['encoder'],
            None,
            lags_observations
        )
        # compute the initial state (posterior at t=0)
        sub_key, key = jax.random.split(key)
        state = self.encode_observations(
            params=params,
            lags_observations=latent_encoder_inputs,
            lags_actions=lags_actions,
            rng=sub_key
        )
        # get prior distributions (predictions) and posterior distributions
        future_latent_states = jax.vmap(self.encoder.apply, in_axes=(None, None, 0))(
            params['encoder'],
            None,
            future_states
        )
        sub_key, key = jax.random.split(key)
        _, trajectory = self.get_trajectory_states(
            params=params,
            rng=sub_key,
            latent_state=future_latent_states,
            actions=actions,
            state=state
        )

        return trajectory

    def predict_trajectory(self, params, state, driver, key):
        """inputs: initial state, driver (trajectory length, -1)
                    state: posterior state (after observing the system at the start of the trajectory)
        compute a trajectory of prior states"""

        traj_mean = []
        traj_std = []
        traj_stoch = []
        traj_deter = []
        for step in range(driver.shape[0]):
            sub_key, key = jax.random.split(key)
            state = self.transition_model.apply(
                params['transition'],
                None,
                state,
                driver[step, :],
                sub_key
            )
            traj_mean.append(state.mean)
            traj_std.append(state.std)
            traj_stoch.append(state.stoch)
            traj_deter.append(state.deter)

        traj_mean = jnp.vstack(traj_mean)
        traj_std = jnp.vstack(traj_std)
        traj_stoch = jnp.vstack(traj_stoch)
        traj_deter = jnp.vstack(traj_deter)

        trajectory = State(mean=traj_mean, deter=traj_deter, stoch=traj_stoch, std=traj_std)

        return trajectory

    def decode_trajectory(self, params, features):
        """features (traj length, dim)"""
        decoded_features = jax.vmap(self.decoder.apply, in_axes=(None, None, 0))(params, None, features)
        return decoded_features

    def encoder(self, params, observations):
        """features (traj length, dim)"""
        latent_observations = jax.vmap(self.encoder.apply, in_axes=(None, None, 0))(params, None, observations)
        return latent_observations

    @staticmethod
    def get_trajectory_features(states: State) -> jnp.array:
        """An element of the state is of size (traj length, -1)"""
        traj_features = jax.vmap(get_features)(states)
        return traj_features

    def get_trajectory_states(self, params, rng, latent_state, actions, state=None):

        """     Apply the observation model multiple times.
                Compute the prior and posterior stochastic state distribution
                parameters: Named Tuple containing the parameters of the networks
                shape of inputs (trajectory length, -1). Batch size must be handled prior to calling this method

                input: state: PriorPostState namedtuple

        """
        if state is None:
            state = PriorPostState(
                prior=initial_state(deter=self.deter, stoch=self.stoch),
                posterior=initial_state(deter=self.deter, stoch=self.stoch)
            )

        def scan_fn(carry, input):
            rng, state = carry
            action, latent_observation = input
            rng, rng1 = jax.random.split(rng)
            state = self.observation_model.apply(
                params['observation'],
                None,
                self.transition_model,
                params['transition'],
                state.posterior,
                action,
                latent_observation,
                rng1
            )
            return (rng, state), PriorPostState(prior=state.prior, posterior=state.posterior)

        (rng, last_state), trajectory_states = jax.lax.scan(scan_fn, (rng, state), (actions, latent_state))

        return last_state, trajectory_states

    def encode_observations(self, params, lags_observations, lags_actions, rng):
        """
        :param params:
        :param encoder_inputs:
        :param rng:
        :return:

        Run the observation model on the observation lags and output the latent state posterior at t=0.
        """

        # Initialize the latent state
        state, _ = self.get_trajectory_states(
            params=params,
            rng=rng,
            latent_state=lags_observations,
            actions=lags_actions
        )

        return state

    @staticmethod
    def get_traj_mode(features: jnp.array):

        def get_log_probability(feature):
            feature_dist = tfp.distributions.Normal(
                loc=feature,
                scale=1.0,
            )
            state = feature_dist.mode()

            return state

        trajectory = jax.vmap(get_log_probability)(features)
        return trajectory

    @staticmethod
    def sample_traj(features: jnp.array, key: jnp.array):

        def get_log_probability(feature, key):
            feature_dist = tfp.distributions.Normal(
                loc=feature,
                scale=1.0,
            )
            state = feature_dist.sample(seed=key)

            return state

        subkeys = jax.random.split(key, features.shape[0])
        trajectory = jax.vmap(get_log_probability, in_axes=(0, 0))(features, subkeys)
        return trajectory
