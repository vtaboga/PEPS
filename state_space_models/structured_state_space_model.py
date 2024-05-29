import jax
import jax.numpy as jnp
import gin
import haiku as hk
import pickle

from state_space_models.base_models.discrete import PhysicsInformedPredictionModel
from state_space_models.abstract_model import ABCModel


@gin.configurable
class StructuredSSM(ABCModel):

    def __init__(
        self,
        seed,
        zone_id,
        observation_size,
        state_size,
        n_lags,
        action_size,
        latent_size,
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
        include_setpoint_change_difference: bool,
        model_output: str = None,
        include_action_lags=False,
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

        self.model = hk.transform(
            lambda x, y: PhysicsInformedPredictionModel(
                encoder_dim=encoder_dim,
                decoder_dim=decoder_dim,
                decoder_depth=decoder_depth,
                hidden_dim=latent_size,
            )(x, y)
        )

        if model_path is not None:
            with open(model_path + '/model_parameters.pkl', "rb") as f:
                self.model_parameters = pickle.load(f)
        else:
            self.model_parameters = None

    def initialize_model(self):
        self.model_key, init_key = jax.random.split(self.model_key)
        dummy_inputs = jnp.ones((self.n_lags, self.encoder_in_size))
        dummy_actions = jnp.ones((self.prediction_horizon, self.action_size))
        model_parameters = self.model.init(init_key, dummy_inputs, dummy_actions)

        return model_parameters

    def run_model(self, params, timestamps, lags_timestamps, inputs, actions, key=None):
        predictions = self.model.apply(
            self.model_parameters,
            None,
            inputs,
            actions,
        )
        return predictions
