import gin
import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

from state_space_models.base_models.ode import GRUODELatentNeuralODE
from state_space_models.abstract_model import ABCModel


@gin.configurable
class CELatentNeuralODE(ABCModel):

    def __init__(
        self,
        seed: int,
        zone_id: int,
        observation_size: int,
        state_size: int,
        action_size: int,
        n_lags: int,
        latent_state_size: int,
        vf_depth: int,
        encoder_dim: int,
        decoder_dim: int,
        decoder_depth: int,
        prediction_horizon: int,
        state_indexes: dict,
        learning_rate: float,
        missing_data_mask: float,
        include_action_lags: bool,
        missing_data_proba: float,
        mean_obs_period: float,
        include_timestep_duration: bool,
        resample_future_states: bool,
        include_setpoint_change_difference:bool,
        model_output: str = None,
        mean=None,
        std=None,
        model_path=None
    ):

        super().__init__(
            model_type='continuous',
            seed=seed,
            zone_id=zone_id,
            observation_size=observation_size,
            state_size=state_size,
            action_size=action_size,
            n_lags=n_lags,
            prediction_horizon=prediction_horizon,
            state_indexes=state_indexes,
            include_action_lags=include_action_lags,
            missing_data_proba=missing_data_proba,
            mean_obs_period=mean_obs_period,
            include_setpoint_change_difference=include_setpoint_change_difference,
            include_timestep_duration=include_timestep_duration,
            resample_future_states=resample_future_states,
            learning_rate=learning_rate,
            model_output=model_output,
            missing_data_mask=missing_data_mask,
            mean=mean,
            std=std
        )

        self.latent_state_size = latent_state_size
        self.vf_depth = vf_depth
        self.vf_hidden_dim = latent_state_size
        self.encoder_dim = encoder_dim
        self.decoder_depth = decoder_depth
        self.seed = seed

        key = jrandom.PRNGKey(seed)
        self.loader_key, self.model_key = jrandom.split(key)

        assert len(self.input_indexes) == self.encoder_in_size * n_lags

        self.model = GRUODELatentNeuralODE(
            latent_space_dim=latent_state_size,
            vf_hidden_dim=latent_state_size,
            output_dim=state_size,
            vf_depth=vf_depth,
            rnn_hidden_size=encoder_dim,
            decoder_depth=decoder_depth,
            decoder_dim=decoder_dim
        )

        self.load_model(model_path)

        @jax.jit
        def jited_model(params, timestamps, lags_timestamps, inputs, actions):
            return self.model(params, timestamps, lags_timestamps, inputs, actions)

        self.jited_model = jited_model

    def initialize_model(self):
        self.model_key, init_enc, init_vf, init_dec = jax.random.split(self.model_key, num=4)

        dummy_inputs = jnp.ones((self.n_lags, self.encoder_in_size + 1))
        dummy_vf_input = jnp.ones(self.vf_hidden_dim + self.action_size + 1)
        dummy_latent_state = jnp.ones(self.vf_hidden_dim)

        encoder_params = self.model.encoder.init(init_enc, dummy_inputs)
        vf_params = self.model.vector_field.init(init_vf, dummy_vf_input)
        decoder_params = self.model.decoder.init(init_dec, dummy_latent_state)

        params = {'encoder': encoder_params, 'vf': vf_params, 'decoder': decoder_params}

        return params

    def run_model(self, params, timestamps, lags_timestamps, inputs, actions, key=None):
        actions_coeffs = diffrax.backward_hermite_coefficients(timestamps, actions)
        predicted_states = self.jited_model(
            self.model_parameters,
            timestamps,
            lags_timestamps,
            inputs,
            actions_coeffs
        )

        return predicted_states
