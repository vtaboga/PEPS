import jax.numpy as jnp
import jax.random as jrandom
import jax
import equinox as eqx
import diffrax

from jax.nn import relu, softplus, tanh


class VectorField(eqx.Module):
    vf_mlp: eqx.nn.MLP
    control_size: int
    latent_state_size: int

    def __init__(self, control_size, latent_state_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.control_size = control_size
        self.latent_state_size = latent_state_size
        self.vf_mlp = eqx.nn.MLP(
            in_size=latent_state_size,
            out_size=latent_state_size * control_size,
            width_size=width_size,
            depth=depth,
            activation=softplus,
            final_activation=tanh,
            key=key
        )

    def __call__(self, t, y, args):
        return self.vf_mlp(y).reshape(self.latent_state_size, self.control_size)


class GRUObservationEncoder(eqx.Module):
    hidden_size: int
    obs_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear

    def __init__(self, obs_size, encoded_obs_size, rnn_hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = rnn_hidden_size
        self.obs_size = obs_size
        self.cell = eqx.nn.GRUCell(obs_size, rnn_hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(rnn_hidden_size, encoded_obs_size, use_bias=False, key=lkey)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = jax.lax.scan(f, hidden, input)
        return self.linear(out)


class InitialEncoder(eqx.Module):
    init_mlp: eqx.nn.MLP

    def __init__(self, control_size, encoded_obs_size, latent_state_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.init_mlp = eqx.nn.MLP(
            in_size=control_size + encoded_obs_size + 1,  # add one for time channel
            out_size=latent_state_size,
            width_size=width_size,
            activation=relu,
            depth=depth,
            key=key
        )

    def __call__(self, obs, control):
        return self.init_mlp(jnp.concatenate([obs, control]))


class Decoder(eqx.Module):
    decoder_mlp: eqx.nn.MLP

    def __init__(self, latent_state_size, state_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.decoder_mlp = eqx.nn.MLP(
            in_size=latent_state_size,
            out_size=state_size,
            width_size=width_size,
            activation=relu,
            depth=depth,
            key=key
        )

    def __call__(self, latent_state):
        return self.decoder_mlp(latent_state)


class NeuralCDE(eqx.Module):
    initial_encoder: InitialEncoder
    obs_encoder: GRUObservationEncoder
    vf: VectorField
    decoder: Decoder
    obs_size: int
    state_size: int

    def __init__(
        self,
        state_size,
        control_size,
        input_observation_size,
        n_lags,
        rnn_hidden_size,
        encoded_obs_size,
        latent_state_size,
        init_encoder_width_size,
        init_encoder_depth,
        model_width_size,
        model_depth,
        decoder_depth,
        model_path,
        *,
        key,
        **kwargs
    ):

        """
        size of the different inputs :

        state_size*n_lags ---[obs encoder]--- encoded_obs_size
                                                     |
                                                     |     ---[init encoder] -- latent_state_size (t0)
                                               control_size                                |
                                                                                           |
                                               control_size---[interpolation]--- [vector field]--[decoder]--state_size
                                                                                           |
                                                                                           |
                                               control_size---[interpolation]--- [vector field]--[decoder]--state_size
                                                                                           |
                                                                                           |


        state_size : size of the state - [Tin, Power]
        control_size : size of the control vector [weather, action]
        n_lags : number of observation lags for the state
        latent_state_size
        """

        super().__init__(**kwargs)
        self.obs_size = state_size * n_lags
        self.state_size = state_size
        state_encoder_key, init_encoder_key, vf_key, decoder_key = jrandom.split(key, 4)
        self.obs_encoder = GRUObservationEncoder(
            obs_size=input_observation_size,
            encoded_obs_size=encoded_obs_size,
            rnn_hidden_size=rnn_hidden_size,
            key=state_encoder_key
        )
        self.initial_encoder = InitialEncoder(
            control_size=control_size - 1,
            encoded_obs_size=encoded_obs_size,
            latent_state_size=latent_state_size,
            width_size=init_encoder_width_size,
            depth=init_encoder_depth,
            key=init_encoder_key
        )
        self.vf = VectorField(
            control_size=control_size,  # add one for time channel
            latent_state_size=latent_state_size,
            width_size=model_width_size,
            depth=model_depth,
            key=vf_key
        )
        self.decoder = Decoder(
            latent_state_size=latent_state_size,
            state_size=self.state_size,
            width_size=latent_state_size // 2,
            depth=decoder_depth,
            key=decoder_key
        )  # predicting Tin and Power
        if model_path is not None:
            self.obs_encoder = eqx.tree_deserialise_leaves(
                model_path + '/observation_encoder.eqx',
                self.obs_encoder
            )
            self.initial_encoder = eqx.tree_deserialise_leaves(
                model_path + '/initial_encoder.eqx',
                self.initial_encoder
            )
            self.vf = eqx.tree_deserialise_leaves(model_path + '/vector_field.eqx', self.vf)
            self.decoder = eqx.tree_deserialise_leaves(model_path + '/decoder.eqx', self.decoder)

    def __call__(self, params, timestamps, lags_timestamps, obs, coeffs):
        control = diffrax.CubicInterpolation(timestamps, coeffs)  # translate inputs to control path X
        term = diffrax.ControlTerm(self.vf, control).to_ode()
        solver = diffrax.Dopri5()
        dt0 = None
        encoded_obs_state = self.obs_encoder(obs)
        init_latent_state = self.initial_encoder(encoded_obs_state, control.evaluate(timestamps[0]))
        saveat = diffrax.SaveAt(ts=timestamps)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            timestamps[0],
            timestamps[-1],
            dt0,
            init_latent_state,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=saveat,
            adjoint=diffrax.RecursiveCheckpointAdjoint()
        )

        prediction = jax.vmap(lambda y: self.decoder(y))(solution.ys)

        return prediction