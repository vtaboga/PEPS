import gin
import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

from typing import Dict
from state_space_models.base_models.ode import NeuralODEPredictionModel
from state_space_models.abstract_model import ABCModel


@gin.configurable
class NeuralODE(ABCModel):

    def __init__(
        self,
        seed: int,
        zone_id: int,
        observation_size: int,
        state_size: int,
        action_size: int,
        n_lags: int,
        latent_state_size: int,
        depth: int,
        prediction_horizon: int,
        include_action_lags: bool,
        missing_data_proba: float,
        mean_obs_period: float,
        include_timestep_duration: bool,
        resample_future_states: bool,
        learning_rate: float,
        missing_data_mask: float,
        include_setpoint_change_difference:bool,
        model_output: str = None,
        state_indexes=None,
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
            include_timestep_duration=include_timestep_duration,
            include_setpoint_change_difference=include_setpoint_change_difference,
            resample_future_states=resample_future_states,
            learning_rate=learning_rate,
            model_output=model_output,
            missing_data_mask=missing_data_mask,
            mean=mean,
            std=std,
        )
        self.latent_state_size = latent_state_size
        self.seed = seed
        key = jrandom.PRNGKey(self.seed)
        self.loader_key, self.model_key, = jrandom.split(key)
        self.model = NeuralODEPredictionModel(
            hidden_dim=latent_state_size,
            output_dim=self.state_size,
            depth=depth,
        )
        self.load_model(model_path)

        @jax.jit
        def jited_model(params, timestamps, lags_timestamps, inputs, actions):
            return self.model(params, timestamps, lags_timestamps, inputs, actions)

        self.jited_model = jited_model

    def initialize_model(self) -> Dict:
        self.model_key, init_vf = jax.random.split(self.model_key)
        dummy_vf_inputs = jnp.ones(self.observation_size + self.action_size + 1)
        vf_params = self.model.vector_field.init(init_vf, dummy_vf_inputs)
        params = {'vf': vf_params}

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
