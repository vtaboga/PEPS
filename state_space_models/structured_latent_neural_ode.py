import gin
import jax
import jax.numpy as jnp
import diffrax

from state_space_models.base_models.ode import LatentPhysicsInformedNeuralODE
from state_space_models.abstract_model import ABCModel


@gin.configurable
class StructuredLatentNeuralODE(ABCModel):

    def __init__(
        self,
        seed,
        zone_id: int,
        observation_size: int,
        state_size: int,
        action_size: int,
        n_lags: int,
        latent_state_size: int,
        vf_depth: int,
        decoder_depth: int,
        encoder_dim: int,
        decoder_dim: int,
        prediction_horizon: int,
        learning_rate: float,
        missing_data_mask: float,
        state_indexes: dict,
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
            include_timestep_duration=include_timestep_duration,
            include_setpoint_change_difference=include_setpoint_change_difference,
            resample_future_states=resample_future_states,
            learning_rate=learning_rate,
            missing_data_mask=missing_data_mask,
            model_output=model_output,
            mean=mean,
            std=std,
        )

        self.latent_state_size = latent_state_size
        self.vf_hidden_dim = latent_state_size

        self.model = LatentPhysicsInformedNeuralODE(
            vf_hidden_dim=latent_state_size,
            output_dim=self.state_size,
            vf_depth=vf_depth,
            decoder_depth=decoder_depth,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.load_model(model_path)

        @jax.jit
        def jited_model(params, timestamps, lags_timestamps, inputs, actions):
            return self.model(params, timestamps, lags_timestamps, inputs, actions)

        self.jited_model = jited_model

    def initialize_model(self):
        self.model_key, init_enc, hvac_init, non_hvac_init, cp_init, eff_init, init_dec = jax.random.split(self.model_key, num=7)

        dummy_inputs = jnp.ones((self.n_lags, self.encoder_in_size))
        dummy_vf_input = jnp.ones(self.vf_hidden_dim + self.action_size + 1)
        dummy_latent_state = jnp.ones(self.vf_hidden_dim)
        dummy_latent_state_entry = jnp.ones(self.vf_hidden_dim // 2)
        encoder_params = self.model.encoder.init(init_enc, dummy_inputs)
        hvac_power_params = self.model.hvac_power.init(hvac_init, dummy_vf_input)
        non_hvac_power_params = self.model.non_hvac_power.init(non_hvac_init, dummy_vf_input)
        decoder_params = self.model.decoder.init(init_dec, dummy_latent_state)
        efficiency_param = self.model.efficiency.init(eff_init, dummy_latent_state_entry)
        cp_param = self.model.Cp.init(cp_init, dummy_latent_state_entry, dummy_latent_state_entry)

        params = {
            'encoder': encoder_params,
            'hvac': hvac_power_params,
            'non_hvac': non_hvac_power_params,
            'decoder': decoder_params,
            'efficiency': efficiency_param,
            'Cp': cp_param
        }

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
