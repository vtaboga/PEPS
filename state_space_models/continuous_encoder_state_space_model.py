import jax
import jax.random as jrandom
import jax.numpy as jnp
import gin
import diffrax

from state_space_models.base_models.ode import GRUODEDiscrete
from state_space_models.abstract_model import ABCModel
from typing import Dict, Union, List


@gin.configurable
class CESSM(ABCModel):

    def __init__(
        self,
        seed,
        zone_id,
        observation_size,
        state_size,
        action_size,
        n_lags,
        latent_state_size,
        encoder_dim,
        decoder_depth,
        decoder_dim,
        prediction_horizon,
        missing_data_proba: float,
        mean_obs_period: float,
        include_timestep_duration: bool,
        resample_future_states: bool,
        learning_rate: float,
        missing_data_mask: float,
        include_setpoint_change_difference:bool,
        model_output: str = None,
        include_action_lags=False,
        state_indexes=None,
        mean=None,
        std=None,
        model_path=None
    ):

        super().__init__(
            model_type='continuous',
            seed=seed,
            zone_id=zone_id,
            state_size=state_size,
            observation_size=observation_size,
            action_size=action_size,
            n_lags=n_lags,
            prediction_horizon=prediction_horizon,
            state_indexes=state_indexes,
            include_action_lags=include_action_lags,
            include_setpoint_change_difference=include_setpoint_change_difference,
            missing_data_proba=missing_data_proba,
            mean_obs_period=mean_obs_period,
            include_timestep_duration=include_timestep_duration,
            resample_future_states=resample_future_states,
            learning_rate=learning_rate,
            model_output=model_output,
            missing_data_mask=missing_data_mask,
            mean=mean,
            std=std,
        )

        key = jrandom.PRNGKey(seed)
        model_key, self.loader_key, self.validation_process_key, self.test_process_key = jrandom.split(key, 4)

        self.latent_state_dim = latent_state_size

        self.model = GRUODEDiscrete(
            latent_space_dim=latent_state_size,
            output_dim=state_size,
            rnn_hidden_size=encoder_dim,
            decoder_depth=decoder_depth,
            decoder_dim=decoder_dim
        )

        self.load_model(model_path)

        @jax.jit
        def jited_model(params, timestamps, lags_timestamps, inputs, actions):
            return self.model(params, timestamps, lags_timestamps, inputs, actions)

        self.jited_model = jited_model

    def initialize_model(self) -> Dict:
        self.model_key, init_enc, init_vf, init_dec = jax.random.split(self.model_key, num=4)

        dummy_inputs = jnp.ones((self.n_lags, self.encoder_in_size + 1))
        dummy_action = jnp.ones(self.action_size)
        dummy_latent_state = jnp.ones(self.latent_state_dim)

        encoder_params = self.model.encoder.init(init_enc, dummy_inputs)
        gru_params = self.model.gru.init(init_vf, dummy_action, dummy_latent_state)
        decoder_params = self.model.decoder.init(init_dec, dummy_latent_state)

        params = {'encoder': encoder_params, 'gru': gru_params, 'decoder': decoder_params}

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
