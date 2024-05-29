import diffrax
import haiku as hk
import jax
import jax.numpy as jnp

from state_space_models.base_models.haiku_networks import LatentRNNEncoder, LatentDecoder, MLP, RNNEncoder
from jax.experimental.ode import odeint
from typing import Optional


class CalorificCapacity(hk.Module):

    def __init__(self, final_activation: str = None, name=None):
        super().__init__(name=name)
        self.final_activation = final_activation

    def __call__(self, non_hvac_power, hvac_power):
        self.Cp = hk.get_parameter("Cp", [], init=hk.initializers.Constant(1.0))
        if self.final_activation is not None:
            if self.final_activation == 'tanh':
                result = jax.nn.tanh(self.Cp * (non_hvac_power + hvac_power))
            else:
                raise ValueError(f'activation function {self.final_activation} not implemented in Cp model')
        else:
            result = self.Cp * (non_hvac_power + hvac_power)
        return result


class ConstantCalorificCapacity(hk.Module):

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, non_hvac_power, hvac_power):
        self.Cp = hk.get_parameter("Cp", [], init=hk.initializers.Constant(1.0))
        result = self.Cp * (non_hvac_power + hvac_power)
        return result


class TemperatureVectorField(hk.Module):

    def __init__(self, final_activation: str = None, name=None):
        super().__init__(name=name)
        self.final_activation = final_activation

    def __call__(self, non_hvac_power, hvac_power):
        if self.final_activation is not None:
            if self.final_activation == 'tanh':
                result = jax.nn.tanh(non_hvac_power + hvac_power)
            else:
                raise ValueError(f'activation function {self.final_activation} not implemented in Cp model')
        else:
            result = non_hvac_power + hvac_power
        return result


class LatentPhysicsInformedNeuralODE:
    def __init__(
        self,
        vf_hidden_dim,
        output_dim,
        vf_depth,
        decoder_depth,
        encoder_dim,
        decoder_dim,
    ):
        self.latent_space_dim = vf_hidden_dim
        self.output_dim = output_dim
        self.encoder = hk.transform(
            lambda x: LatentRNNEncoder(vf_hidden_dim, encoder_dim)(x)
        )
        self.hvac_power = hk.transform(
            lambda x: MLP(vf_hidden_dim // 2, vf_depth, vf_hidden_dim // 2, is_vector_field=True)(x)
        )
        self.non_hvac_power = hk.transform(
            lambda x: MLP(vf_hidden_dim // 2, vf_depth, vf_hidden_dim // 2, is_vector_field=True)(x)
        )
        self.efficiency = hk.transform(
            lambda x: x @ hk.get_parameter(
                "efficiency_matrix",
                shape=(vf_hidden_dim // 2, vf_hidden_dim // 2),
                init=hk.initializers.RandomNormal()
            )
        )
        self.Cp = hk.transform(lambda x, y: CalorificCapacity(final_activation='tanh')(x, y))
        self.decoder = hk.transform(
            lambda x: LatentDecoder(vf_hidden_dim, decoder_depth, decoder_dim)(x)
        )

    def __call__(self, params, timestamps, lags_timestamps, inputs, action_coeffs):
        """
            params: parameters of the neural network (vector field)
            timestamps: arrays of float times for evaluation
            state: state of the system at t=0
            action: action from t=0 to t=T-1
        """

        dt = timestamps[-1] - timestamps[-2]
        # add an extra timestep because odeint return the initial state as first element and not the first prediction
        timestamps = jnp.append(timestamps, timestamps[-1] + dt)
        actions = diffrax.CubicInterpolation(timestamps[1:], action_coeffs)

        @jax.jit
        def apply_net(encoded_state, t, actions):
            inputs = jnp.concatenate([jnp.array([t]), encoded_state, actions.evaluate(t)], axis=-1)
            hvac = self.hvac_power.apply(params['hvac'], None, inputs)
            hvac_heat = self.efficiency.apply(params['efficiency'], None, hvac)
            non_hvac = self.non_hvac_power.apply(params['non_hvac'], None, inputs)
            temperature = self.Cp.apply(params['Cp'], None, non_hvac, hvac_heat)
            return jnp.concatenate([hvac, temperature])

        encoded_states = self.encoder.apply(params['encoder'], None, inputs)
        trajectory = odeint(apply_net, encoded_states, timestamps, actions)
        trajectory = trajectory[1:, :]  # only keep future states
        states = jax.vmap(self.decoder.apply, in_axes=(None, None, 0))(params['decoder'], None, trajectory)
        return states


class LatentNeuralODEPredictionModel:
    def __init__(
        self,
        latent_space_dim,
        vf_hidden_dim,
        output_dim,
        vf_depth,
        rnn_hidden_size,
        decoder_depth,
        decoder_dim
    ):
        self.latent_space_dim = latent_space_dim
        self.output_dim = output_dim
        self.encoder = hk.transform(
            lambda x: RNNEncoder(encoded_obs_size=latent_space_dim, rnn_hidden_size=rnn_hidden_size)(
                x
            )
        )
        self.vector_field = hk.transform(
            lambda x: MLP(vf_hidden_dim, vf_depth, vf_hidden_dim, is_vector_field=True)(x)
        )
        self.decoder = hk.transform(
            lambda x: MLP(output_dim, decoder_depth, decoder_dim, is_vector_field=False)(x)
        )

    def __call__(self, params, timestamps, lags_timestamps, state, action_coeffs):
        """
            params: parameters of the neural network (vector field)
            timestamps: arrays of float times for evaluation
            state: state of the system at t=0
        """
        dt = timestamps[-1] - timestamps[-2]
        # add an extra timestep because odeint return the initial state as first element and not the first prediction
        timestamps = jnp.append(timestamps, timestamps[-1] + dt)
        actions = diffrax.CubicInterpolation(timestamps[1:], action_coeffs)

        @jax.jit
        def apply_net(state, t, actions, params):
            action = actions.evaluate(t)
            inputs = jnp.concatenate([jnp.array([t]), state, action], axis=-1)
            res = self.vector_field.apply(params, None, inputs)
            return res

        encoded_state = self.encoder.apply(params['encoder'], None, state)
        trajectory = odeint(apply_net, encoded_state, timestamps, actions, params['vf'])
        trajectory = trajectory[1:, :]
        states = jax.vmap(self.decoder.apply, in_axes=(None, None, 0))(params['decoder'], None, trajectory)
        return states


class NeuralODEPredictionModel:
    def __init__(self, hidden_dim, output_dim, depth):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vector_field = hk.transform(lambda x: MLP(output_dim, depth, hidden_dim, is_vector_field=True)(x))

    def __call__(self, params, timestamps, lags_timestamps, state, actions_coeffs):
        """
            params: parameters of the neural network (vector field)
            timestamps: arrays of float times for evaluation
            state: state of the system at t=0
            action: continuous interpolation of the actions
        """

        dt = timestamps[-1] - timestamps[-2]
        # add an extra timestep because odeint return the initial state as first element and not the first prediction
        timestamps = jnp.append(timestamps, timestamps[-1] + dt)
        actions = diffrax.CubicInterpolation(timestamps[1:], actions_coeffs)
        last_state = state[-1, :]  # corresponds to the system state at t=0

        @jax.jit
        def apply_net(state, t, actions):
            inputs = jnp.concatenate([jnp.array([t]), state, actions.evaluate(t)], axis=-1)
            out = self.vector_field.apply(params['vf'], None, inputs)
            return out

        trajectory = odeint(apply_net, last_state, timestamps, actions, rtol=1e-6, atol=1e-6)
        trajectory = trajectory[1:, :]

        return trajectory


class PhysicsInformedNeuralODE:

    def __init__(self, latent_state_size, output_dim, depth):
        self.hvac_power = hk.transform(
            lambda x: MLP(
                output_dim=output_dim // 2,
                depth=depth,
                hidden_dim=latent_state_size // 2,
                is_vector_field=True
            )(x)
        )
        self.non_hvac_power = hk.transform(
            lambda x: MLP(
                output_dim=output_dim // 2,
                depth=depth,
                hidden_dim=latent_state_size // 2,
                is_vector_field=True
            )(x)
        )
        self.Cp = hk.transform(lambda x, y: ConstantCalorificCapacity()(x, y))

    def __call__(self, params, timestamps, lags_timestamps, inputs, action_coeffs):
        action = diffrax.CubicInterpolation(timestamps, action_coeffs)
        last_state = inputs[-1, :]

        @jax.jit
        def apply_net(state, t, action):
            inputs = jnp.concatenate([jnp.array([t]), state, action.evaluate(t)], axis=-1)
            hvac = self.hvac_power.apply(params['hvac'], None, inputs)
            non_hvac = self.non_hvac_power.apply(params['non_hvac'], None, inputs)
            temperature = self.Cp.apply(params['cp'], None, hvac, non_hvac)
            return jnp.concatenate([hvac, temperature])

        trajectory = odeint(apply_net, last_state, timestamps, action, rtol=1e-6, atol=1e-6)

        return trajectory

class GRUODE:

    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        w_i_init: Optional[hk.initializers.Initializer] = None,
        w_h_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None
    ):
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.w_i_init = w_i_init
        self.w_h_init = w_h_init
        self.b_init = b_init
        self.name = name

        self.dynamics_gru = hk.transform(
            lambda x, h: hk.GRU(
                hidden_size=hidden_size,
                w_i_init=w_i_init,
                w_h_init=w_h_init,
                b_init=b_init
            )(x, h)
        )
        self.core_gru = hk.transform(
            lambda x, h: hk.GRU(
                hidden_size=hidden_size,
                w_i_init=w_i_init,
                w_h_init=w_h_init,
                b_init=b_init
            )(x, h)
        )

        self.linear = hk.transform(
            lambda x: hk.Linear(output_size=output_size)(x)
        )

    def init(self, rng, value: jax.Array):
        """
        Initialises the `GRUODE` module.

        arguments:
            rng  : a pseudo-random number generator (PRNG) key.
            value: any arbitrary value for the inputs of the module.

        returns:
            a mapping (dict) of inner module identifiers (names) to their respective
            initial parameter values.

        """

        dummy_hidden = jnp.zeros((self.hidden_size))

        dynamics_rng_key, core_rng_key, linear_rng_key = jax.random.split(rng, num=3)

        dynamics_gru_params = self.dynamics_gru.init(
            dynamics_rng_key,
            jnp.expand_dims(value[..., 0], axis=-1), dummy_hidden
        )
        core_gru_params = self.core_gru.init(
            core_rng_key,
            value[..., 1:], dummy_hidden
        )
        linear_params = self.linear.init(
            linear_rng_key,
            dummy_hidden
        )

        return {
            "dynamics_gru": dynamics_gru_params,
            "core_gru": core_gru_params,
            "linear": linear_params
        }

    def apply(self, params, rng, sequence: jax.Array):
        """
        Applies the `GRUODE` module to a sequence of input values.

        arguments:
            params  : value for the parameters of the module.
            rng     : a pseudo-random number generator (PRNG) key.
            sequence: an array of equentially observed values for the inputs `( (t_0, x_0),
                                                                                (t_1, x_1),
                                                                                ...,
                                                                                (t_i, x_i),
                                                                                ...
                                                                                (t_n, x_n) )`.

        returns:
            the ouputs resulting from applying the module to the provide sequence of inputs.

        """

        def init_fn(inputs):
            """
            Applies the `core-gru` to the first value in the input sequence (initial value),
            in order to provide a correspondig initial value for the [hidden] state variable
            of the `GRUODE`.

            argumrnts:
                inputs: initial value for the observable inputs `x_0`.

            returns:
                the initial value for the [hidden] state variable of the `GRUODE`.

            """

            t_init, x_init = inputs[0], inputs[1:]
            h_init, _ = self.core_gru.apply(
                params["core_gru"], None,
                x_init, jnp.zeros((self.hidden_size))
            )

            state_init = jnp.concatenate((jnp.array([t_init]), h_init), axis=-1)
            return state_init

        @jax.jit
        def core_fn(state, inputs):
            """
            Applies the following 2-stade-procedure:
            1. integrates the gru dynamics forward from the provided [hidden] state value,
            2. applies the `core-gru` to adapt the terminal value.

            arguments:
                state : previous value for the [hidden] state `(t_{i-1}, h_{i-1})`.
                inputs: current value for the observable inputs `(t_i, x_i)`.

            returns:
                the current value for the [hidden] state `(t_i, h_i)`.

            """

            t_curr, x_curr = inputs[0], inputs[1:]
            t_prev, h_prev = state[0], state[1:]

            def gru_dynamics_fn(h, t):
                v, _ = self.dynamics_gru.apply(
                    params["dynamics_gru"], None,
                    jnp.array([t]), h
                )
                return v - h

            _, h_prime = odeint(gru_dynamics_fn, h_prev, jnp.array([t_prev, t_curr]))
            h_curr, _ = self.core_gru.apply(
                params["core_gru"], None,
                x_curr, h_prime
            )

            state_curr = jnp.concatenate((jnp.array([t_curr]), h_curr), axis=-1)

            return state_curr, state_curr

        state_term, _ = jax.lax.scan(core_fn, init_fn(sequence[0]), sequence[1:])
        h_term = state_term[1:]
        outputs = self.linear.apply(params["linear"], None, h_term)

        return outputs


class GRUODELatentNeuralODE:

    def __init__(
        self,
        latent_space_dim,
        vf_hidden_dim,
        output_dim,
        vf_depth,
        rnn_hidden_size,
        decoder_depth,
        decoder_dim
    ):
        self.latent_space_dim = latent_space_dim
        self.output_dim = output_dim

        self.encoder = GRUODE(
            output_size=latent_space_dim, hidden_size=rnn_hidden_size
        )
        self.vector_field = hk.transform(
            lambda x: MLP(vf_hidden_dim, vf_depth, vf_hidden_dim, is_vector_field=True)(x)
        )
        self.decoder = hk.transform(
            lambda x: MLP(output_dim, decoder_depth, decoder_dim, is_vector_field=False)(x)
        )

    def __call__(self, params, timestamps, lags_timestamps, state, action_coeffs):
        """
            params: parameters of the neural network (vector field)
            timestamps: arrays of float times for evaluation
            state: state of the system at t=0
            driver: driver from t=0 to t=T-1
        """

        dt = timestamps[-1] - timestamps[-2]
        # add an extra timestep because odeint return the initial state as first element and not the first prediction
        timestamps = jnp.append(timestamps, timestamps[-1] + dt)
        actions = diffrax.CubicInterpolation(timestamps[1:], action_coeffs)

        @jax.jit
        def apply_net(state, t, actions, params):
            inputs = jnp.concatenate([jnp.array([t]), state, actions.evaluate(t)], axis=-1)
            res = self.vector_field.apply(params, None, inputs)
            return res

        state = jnp.concatenate(
            (jnp.expand_dims(lags_timestamps, axis=-1), state),
            axis=-1
        )
        encoded_states = self.encoder.apply(params['encoder'], None, state)
        trajectory = odeint(apply_net, encoded_states, timestamps, actions, params['vf'])
        trajectory = trajectory[1:, :]
        states = jax.vmap(self.decoder.apply, in_axes=(None, None, 0))(params['decoder'], None, trajectory)
        return states


class GRUODEDiscrete:

    def __init__(
        self,
        latent_space_dim,
        output_dim,
        rnn_hidden_size,
        decoder_depth,
        decoder_dim,
    ):
        self.latent_space_dim = latent_space_dim
        self.output_dim = output_dim

        self.encoder = GRUODE(
            output_size=latent_space_dim,
            hidden_size=rnn_hidden_size
        )
        self.gru = hk.transform(lambda x, y: hk.GRU(hidden_size=latent_space_dim)(x, y))
        self.decoder = hk.transform(
            lambda x: MLP(
                depth=decoder_depth,
                hidden_dim=decoder_dim,
                output_dim=output_dim
            )(x)
        )

    def __call__(self, params, timestamps, lags_timestamps, state, actions_coeffs):

        actions_interpolation = diffrax.CubicInterpolation(timestamps, actions_coeffs)
        actions = jax.vmap(actions_interpolation.evaluate)(timestamps)

        def scan_fn(carry, input):
            _, new_state = self.gru.apply(params['gru'], None, input, carry)
            return new_state, new_state

        state = jnp.concatenate(
            (jnp.expand_dims(lags_timestamps, axis=-1), state),
            axis=-1
        )
        latent_state = self.encoder.apply(
            params=params['encoder'],
            rng=None,
            sequence=state
        )
        _, trajectory = jax.lax.scan(scan_fn, latent_state, actions)
        decoded_trajectory = jax.vmap(self.decoder.apply, in_axes=(None, None, 0))(
            params['decoder'],
            None,
            trajectory
        )

        return decoded_trajectory
