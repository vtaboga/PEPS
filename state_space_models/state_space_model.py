import jax
import jax.random as jrandom
import jax.numpy as jnp
import gin
import haiku as hk

from state_space_models.base_models.discrete import PredictionModel
from state_space_models.abstract_model import ABCModel


@gin.configurable
class SSM(ABCModel):

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
        model_output: str = None,
        include_action_lags=False,
        include_setpoint_change_difference=False,
        state_indexes=None,
        mean=None,
        std=None,
        model_path=None
    ):

        super().__init__(
            model_type='discrete',
            seed=seed,
            zone_id=zone_id,
            observation_size=observation_size,
            state_size=state_size,
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

        print("--- SSM ---")
        print(f"decoder_depth: {decoder_depth}")
        print(f"decoder_dim: {decoder_dim}")
        print(f"latent_state_size: {latent_state_size}")
        print(f"encoder_dim: {encoder_dim}")
        print(f"state_size: {state_size}")
        print(f"model_output: {model_output}")
        print(f"state_indexes: {state_indexes}")
        print(f"model_path: {model_path}")
        print(f"n_lags: {n_lags}")
        print(f"state_size: {state_size}")
        print(f"action_size: {action_size}")

        key = jrandom.PRNGKey(seed)
        model_key, self.loader_key, self.validation_process_key, self.test_process_key = jrandom.split(key, 4)

        self.model = hk.transform(
            lambda x, y:
            PredictionModel(
                decoder_depth=decoder_depth,
                decoder_dim=decoder_dim,
                hidden_dim=latent_state_size,
                encoder_dim=encoder_dim,
                output_dim=state_size
            )(x, y)
        )

        @jax.jit
        def jited_model(params, timestamps, lags_timestamps, inputs, actions):
            return self.model.apply(params, None, inputs, actions)

        self.jited_model = jited_model

        self.load_model(model_path)

    def initialize_model(self):
        self.model_key, init_key = jax.random.split(self.model_key)
        dummy_inputs = jnp.ones((self.n_lags, self.encoder_in_size))
        dummy_actions = jnp.ones((self.prediction_horizon, self.action_size))
        model_parameters = self.model.init(init_key, dummy_inputs, dummy_actions)

        return model_parameters

    def run_model(self, params, timestamps, lags_timestamps, inputs, actions, key=None):
        predictions = self.jited_model(
            params,
            timestamps,
            lags_timestamps,
            inputs,
            actions,
        )
        return predictions
