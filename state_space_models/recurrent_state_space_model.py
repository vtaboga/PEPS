import jax
import jax.numpy as jnp
import optax
import gin
import numpy as np

from tensorflow_probability.substrates import jax as tfp
from typing import Dict
from state_space_models.base_models.rssm import initial_state, get_distribution, RecurrentStateSpacePredictionModel
from state_space_models.abstract_model import ABCModel


@gin.configurable
class RSSM(ABCModel):

    def __init__(
        self,
        seed,
        zone_id,
        observation_size,
        state_size,
        action_size,
        n_lags,
        encoder_depth,
        encoder_activation,
        decoder_depth,
        decoder_activation,
        transition_model_activation,
        determinisitc_size,
        stochastic_size,
        latent_state_size,
        prediction_horizon,
        missing_data_proba: float,
        mean_obs_period: float,
        include_timestep_duration: bool,
        resample_future_states: bool,
        learning_rate: float,
        loss_kl_scale: float,
        missing_data_mask: float,
        include_setpoint_change_difference:bool,
        include_action_lags=False,
        state_indexes=None,
        mean=None,
        std=None,
        model_path=None
    ):

        self.observed_state_size = observation_size  # The observation size will include the lags of actions if any
        model_output = None  # RSSM only work with predictions of both temperature and power
        super().__init__(
            model_type='stochastic',
            zone_id=zone_id,
            seed=seed,
            state_size=state_size,
            action_size=action_size,
            observation_size=observation_size,
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
        self.loss_kl_scale = loss_kl_scale

        key = jax.random.PRNGKey(seed)
        self.key, self.obs_key, self.decoder_key, self.encoder_key, self.transition_key = jax.random.split(
            key,
            num=5
        )

        self.model = RecurrentStateSpacePredictionModel(
            state_size=self.state_size,
            action_size=self.action_size,
            observed_state_size=self.observed_state_size,
            encoder_depth=encoder_depth,
            encoder_activation=encoder_activation,
            latent_state_size=latent_state_size,
            decoder_activation=decoder_activation,
            decoder_depth=decoder_depth,
            stochastic_size=stochastic_size,
            deterministic_size=determinisitc_size,
            transition_model_activation=transition_model_activation
        )

        self.load_model(model_path)

        @jax.jit
        def jited_model_mode(params, timestamps, lags_timestamps, inputs, actions, key):
            return self.model(params, timestamps, lags_timestamps, inputs, actions, key)

        @jax.jit
        def jited_model_sample(params, timestamps, lags_timestamps, inputs, actions, key):
            return self.model.sample_predictions(params, timestamps, lags_timestamps, inputs, actions, key)

        self.jited_model_mode = jited_model_mode
        self.jited_model_sample = jited_model_sample

    def initialize_model(self) -> Dict:

        self.decoder_key, init_decoder_key = jax.random.split(self.decoder_key)
        self.encoder_key, init_encoder_key = jax.random.split(self.encoder_key)
        self.obs_key, init_obs_key = jax.random.split(self.obs_key)
        self.transition_key, init_transition_key = jax.random.split(self.transition_key)
        dummy_latent_state = initial_state(deter=self.model.deter, stoch=self.model.stoch)
        dummy_latent_obs = jnp.ones(self.model.latent_state_size)
        dummy_obs = jnp.ones(self.observed_state_size)
        dummy_actions = jnp.ones(self.action_size)
        init_transition_key1, init_transition_key2 = jax.random.split(init_transition_key)
        transition = self.model.transition_model.init(
            init_transition_key1,
            dummy_latent_state,
            dummy_actions,
            init_transition_key2
        )
        init_obs_key1, init_obs_key2 = jax.random.split(init_obs_key)
        obs = self.model.observation_model.init(
            init_obs_key1,
            self.model.transition_model,
            transition,
            dummy_latent_state,
            dummy_actions,
            dummy_latent_obs,
            init_obs_key2
        )
        decoder = self.model.decoder.init(init_decoder_key, jnp.ones(self.model.deter + self.model.stoch))
        encoder = self.model.encoder.init(init_encoder_key, dummy_obs)
        params = {'observation': obs, 'decoder': decoder, 'transition': transition, 'encoder': encoder}
        return params

    def process_lags(self, prediction_horizon, energyplus_timestep_duration):
        lags_timestamps = jnp.linspace(0, (self.n_lags - 1), self.n_lags) * energyplus_timestep_duration
        min_lag, max_lag = jnp.min(lags_timestamps), jnp.max(lags_timestamps)
        lags_timestamps = (lags_timestamps - min_lag) / (max_lag - min_lag)
        timestamps = jnp.linspace(0, 1, prediction_horizon)

        return lags_timestamps, timestamps

    def make_predictions(self, inputs, actions, energyplus_timestep_duration, prediction_horizon, key):
        """
        :param inputs: normalized encoder inputs of shape (n_lags, observation dim)
        :param actions:  normalized actions of shape (horizon, action_dim)
        :param energyplus_timestep_duration: time step of the simulation in minutes
        :param prediction_horizon: number of time steps to predict
        :return: denormalized predictions in shape (horizon, n observations). Usually (power, temperature)
        """

        lags_timestamps, timestamps = self.process_lags(prediction_horizon, energyplus_timestep_duration)
        predictions = self.run_model(self.model_parameters, timestamps, lags_timestamps, inputs, actions, key)
        predictions = self.normalizer.denormalize_predictions(predictions)

        return predictions

    def sample_predictions(self, inputs, actions, energyplus_timestep_duration, prediction_horizon, key):
        lags_timestamps, timestamps = self.process_lags(prediction_horizon, energyplus_timestep_duration)
        predictions = self.sample_from_model(
                self.model_parameters,
                timestamps,
                lags_timestamps,
                inputs,
                actions,
                key
            )
        predictions = self.normalizer.denormalize_predictions(predictions)

        return predictions

    def run_model(self, params, timestamps, lags_timestamps, inputs, actions, key):
        predicted_states = self.jited_model_mode(
            self.model_parameters,
            timestamps,
            lags_timestamps,
            inputs,
            actions,
            key
        )

        return predicted_states

    def sample_from_model(self, params, timestamps, lags_timestamps, inputs, actions, key):
        predicted_states = self.jited_model_sample(
            params,
            timestamps,
            lags_timestamps,
            inputs,
            actions,
            key
        )

        return predicted_states

    def train_model(
        self,
        training_data,
        validation_data,
        loss_temperature_weight,
        training_batch_size,
        validation_batch_size,
        training_steps: int,
        early_stop_lag=2,
        validation_every=500,
        early_stopping=False,
        model_path: str = None
    ):

        @jax.jit
        def make_batch_predictions(params, batch, key):
            """make predictions for an odered batch of shape (batch_length, trajectory_length, state_size)
            the predictions are made using jax vmap and put back in the right order afterwards"""

            lags_timestamps, encoder_inputs, timestamps, actions, future_states = batch
            batch_size = future_states.shape[0]
            sub_keys = jax.random.split(key, num=batch_size)
            predicted_states = jax.vmap(self.model, in_axes=(None, None, 0, 0, 0, 0))(
                params,
                timestamps,
                lags_timestamps,
                encoder_inputs,
                actions,
                sub_keys
            )
            predicted_states = predicted_states.reshape((batch_size * self.prediction_horizon), self.state_size)
            states = future_states.reshape((batch_size * self.prediction_horizon), self.state_size)

            return predicted_states, states

        def validation_step(params, validation_data, batch_size, key):
            predictions = []
            targets = []
            for step, (val_batch,) in enumerate(self._dataloader_test((validation_data,), batch_size, key=None)):
                val_batch = self.process_batch(val_batch, self.validation_process_key)
                if val_batch.encoder_inputs is None or val_batch.future_states is None:
                    continue
                predicted_states, states = make_batch_predictions(params, val_batch, key)
                predictions.append(predicted_states)
                targets.append(states)

            total_length = int(len(predictions) * predictions[0].shape[0])
            predictions = jnp.array(predictions).reshape((total_length, self.state_size))
            targets = jnp.array(targets).reshape((total_length, self.state_size))

            if self.state_size == 2:
                hvac_power_targets = self.normalizer.denormalize_hvac_power(targets[:, 0])
                hvac_power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(predictions[:, 0]), 0)
                mae_hvac_power = jnp.mean(jnp.abs((hvac_power_targets - hvac_power_preds)))
                temperature_targets = self.normalizer.denormalize_temperature(targets[:, 1])
                temperature_preds = self.normalizer.denormalize_temperature(predictions[:, 1])
                mae_temperature = jnp.mean(jnp.abs((temperature_targets - temperature_preds)))
            elif self.model_output == 'hvac_power':
                hvac_power_targets = self.normalizer.denormalize_hvac_power(targets[:, 0])
                hvac_power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(predictions[:, 0]), 0)
                mae_hvac_power = jnp.mean(jnp.abs((hvac_power_targets - hvac_power_preds)))
                temperature_targets = self.normalizer.denormalize_temperature(jnp.zeros(targets[:, 1].shape))
                temperature_preds = self.normalizer.denormalize_temperature(jnp.zeros(targets[:, 1].shape))
                mae_temperature = jnp.mean(jnp.abs((temperature_targets - temperature_preds)))
            else:
                temperature_targets = self.normalizer.denormalize_temperature(targets[:, 1])
                temperature_preds = self.normalizer.denormalize_temperature(predictions[:, 1])
                mae_temperature = jnp.mean(jnp.abs((temperature_targets - temperature_preds)))
                hvac_power_targets = self.normalizer.denormalize_hvac_power(jnp.zeros(targets[:, 0].shape))
                hvac_power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(jnp.zeros(targets[:, 0].shape)), 0)
                mae_hvac_power = jnp.mean(jnp.abs((hvac_power_targets - hvac_power_preds)))

            return mae_hvac_power, mae_temperature

        @jax.jit
        def loss_fn(params, timestamps, lags_timestamps, inputs, future_states, actions, key):
            """Operations :
            observation(observation, driver) -> prior trajectories, posterior trajectories
            get features (prior trajectories) -> prior state features
            zone obs log probability -> get log prob (prior state features)
            kl_divergence(prior, posterior)

            loss = kl_divergence - zone obs log prob
            """

            batch_size = future_states.shape[0]
            sub_keys = jax.random.split(key, num=batch_size)
            trajectories = jax.vmap(self.model.call_training, in_axes=(None, None, 0, 0, 0, 0, 0))(
                params,
                timestamps,
                lags_timestamps,
                inputs,
                future_states,
                actions,
                sub_keys
            )
            prior_states, posterior_states = trajectories.prior, trajectories.posterior
            # Compute observation probability using the posterior distribution
            features = jax.vmap(self.model.get_trajectory_features, in_axes=(0,))(posterior_states)
            features = jax.vmap(self.model.decode_trajectory, in_axes=(None, 0))(params['decoder'], features)
            zone_state_probability = jax.vmap(self.get_traj_log_probability)(features, future_states)
            obs_loss = jnp.mean(zone_state_probability)
            # Compute kl divergence between prior and posterior distributions
            kl_divergences = jax.vmap(self.traj_kl_divergence)(posterior_states, prior_states)
            kl_div = jnp.mean(kl_divergences)
            model_loss = self.loss_kl_scale * kl_div - obs_loss
            return model_loss

        @jax.jit
        def training_step(params, batch, opt_state, key):
            lags_timestamps, encoder_inputs, timestamps, actions, future_states = batch
            key, new_key = jax.random.split(key)
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(params, timestamps, lags_timestamps, encoder_inputs, future_states, actions, key)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_opt_state, new_params, loss, new_key

        self.key, key, loader_key = jax.random.split(self.key, num=3)
        print(f'------ training for {training_steps} steps ------')
        if self.model_parameters is None:
            params = self.initialize_model()
        else:
            params = self.model_parameters
        training_logs = {'n_steps': [], 'hvac_power': [], 'temperature': [], 'mean_train_loss': []}
        optimizer = optax.adabelief(self.learning_rate)
        opt_state = optimizer.init(params)
        losses = [0]
        for step, (minibatch,) in zip(
                range(training_steps),
                self._dataloader_training((training_data,), training_batch_size, key=self.loader_key)
        ):
            batch_key, training_key, self.model_key = jax.random.split(self.model_key, num=3)
            minibatch = self.process_batch(minibatch, key=training_key)
            if not (minibatch.encoder_inputs is None or minibatch.future_states is None):
                opt_state, params, loss, key = training_step(
                    params,
                    minibatch,
                    opt_state,
                    batch_key
                )
                losses.append(loss)
            else:
                loss = 0.0

            if (validation_data is not None) and ((step % validation_every) == 0 or step == training_steps - 1):
                validation_key, self.model_key = jax.random.split(self.model_key)
                mae_hvac_power, mae_temperature = validation_step(
                    params,
                    validation_data,
                    validation_batch_size,
                    key=validation_key
                )
                training_logs['hvac_power'].append(mae_hvac_power)
                training_logs['temperature'].append(mae_temperature)
                training_logs['n_steps'].append(step)
                training_logs['mean_train_loss'].append(np.mean(losses))
                print(
                    f'MAE validation : hvac_power {np.round(mae_hvac_power, 4)} kW, '
                    f'temperature {np.round(mae_temperature, 4)} degree C'
                )

                if model_path is not None:
                    self.save_model(path=model_path, final=False)

                # early stoping on hvac_power
                if self.model_output == 'hvac_power':
                    if early_stopping and (len(training_logs['hvac_power']) > early_stop_lag + 1) and (
                            training_logs['hvac_power'][-1] > max(
                        training_logs['hvac_power'][-early_stop_lag - 1:-1]
                    )):
                        break

        self.model_parameters = params
        self._training_logs = training_logs

    @staticmethod
    def get_traj_log_probability(features: jnp.array, states: jnp.array):

        def get_log_probability(feature, state):
            feature_dist = tfp.distributions.Normal(
                loc=feature,
                scale=1.0,
            )
            state_probability = feature_dist.log_prob(state)

            return state_probability

        traj_log_prob = jax.vmap(get_log_probability)(features, states)
        return traj_log_prob

    @staticmethod
    def traj_kl_divergence(posterior_states, prior_states):

        def get_kl_divergence(prior_state, posterior_state):
            prior_dist = get_distribution(prior_state)
            posterior_dist = get_distribution(posterior_state)
            return tfp.distributions.kl_divergence(posterior_dist, prior_dist)

        traj_kl_div = jax.vmap(get_kl_divergence)(prior_states, posterior_states)
        return traj_kl_div
