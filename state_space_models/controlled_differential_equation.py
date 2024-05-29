import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax
import optax
import equinox as eqx
import gin
import os

from typing import Dict
from state_space_models.base_models.cde import NeuralCDE
from state_space_models.abstract_model import ABCModel


@gin.configurable
class CDE(ABCModel):
    """
    The training method is re-implemented because we use Diffrax to implement CDEs
    """

    def __init__(
        self,
        seed: int,
        zone_id: int,
        state_size: int,
        observation_size: int,
        action_size: int,
        n_lags: int,
        encoded_obs_size: int,
        latent_state_size: int,
        rnn_hidden_size: int,
        model_depth: int,
        model_width_size: int,
        init_encoder_depth: int,
        init_encoder_width_size: int,
        decoder_depth: int,
        prediction_horizon: int,
        state_indexes: dict,
        include_action_lags: bool,
        missing_data_proba: float,
        mean_obs_period: float,
        include_timestep_duration: bool,
        resample_future_states: bool,
        learning_rate: float,
        missing_data_mask: float,
        include_setpoint_change_difference:bool,
        model_output: str = None,
        mean=None,
        std=None,
        model_path=None
    ):
        """
        :param seed:
        :param state_size:
        :param observation_size:
        :param action_size:
        :param n_lags:
        :param encoded_obs_size:
        :param latent_state_size:
        :param rnn_hidden_size:
        :param model_depth:
        :param model_width_size:
        :param init_encoder_depth:
        :param init_encoder_width_size:
        :param decoder_depth:
        :param prediction_horizon:
        :param state_indexes:
        :param include_action_lags:
        :param missing_data_proba:
        :param mean_obs_period:
        :param include_timestep_duration:
        :param resample_future_states:
        :param learning_rate:
        :param missing_data_mask:
        :param model_output:
        :param mean:
        :param std:
        :param model_path:
        """

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
            include_setpoint_change_difference=include_setpoint_change_difference,
            mean_obs_period=mean_obs_period,
            include_timestep_duration=include_timestep_duration,
            resample_future_states=resample_future_states,
            learning_rate=learning_rate,
            missing_data_mask=missing_data_mask,
            model_output=model_output,
            mean=mean,
            std=std
        )
        self.prediction_horizon = prediction_horizon
        self.state_size = state_size
        self.n_lags = n_lags
        self.obs_size = state_size * n_lags
        self.latent_state_size = latent_state_size
        self.seed = seed
        key = jrandom.PRNGKey(self.seed)
        self.model_key, self.loader_key, self.data_key = jrandom.split(key, 3)

        self.model = NeuralCDE(
            state_size=state_size,
            control_size=action_size,
            input_observation_size=observation_size,
            n_lags=n_lags,
            encoded_obs_size=encoded_obs_size,
            rnn_hidden_size=rnn_hidden_size,
            latent_state_size=latent_state_size,
            init_encoder_width_size=init_encoder_width_size,
            init_encoder_depth=init_encoder_depth,
            model_width_size=model_width_size,
            model_depth=model_depth,
            decoder_depth=decoder_depth,
            key=self.model_key,
            model_path=model_path
        )

        self.model_parameters = {
            'obs_encoder': self.model.obs_encoder,
            'initial_encoder': self.model.initial_encoder,
            'vf': self.model.vf,
            'decoder': self.model.decoder
        }

        @eqx.filter_jit
        def jited_model(timestamps, lags_timestamps, state, action_coeffs):
            return self.model(None, timestamps, lags_timestamps, state, action_coeffs)

        self.jited_model = jited_model

    def initialize_model(self) -> Dict:
        """weights are stored directly in the model. Save model and load model are overwritten"""
        return dict()

    def train_model(
        self,
        training_data,
        validation_data,
        loss_temperature_weight,
        validation_batch_size,
        training_batch_size,
        training_steps: int,
        early_stop_lag=2,
        validation_every=500,
        early_stopping=False,
        model_path: str = None
    ):

        @eqx.filter_jit
        def make_batch_predictions(model, batch):
            """make predictions for an odered batch of shape (batch_length, trajectory_length, state_size)
                the predictions are made using jax vmap and put back in the right order afterwards"""
            lags_timestamps, encoder_inputs, timestamps, actions, future_states = batch
            batch_size = future_states.shape[0]
            actions_coeffs = jax.vmap(diffrax.backward_hermite_coefficients, in_axes=(None, 0))(
                timestamps,
                actions
            )
            predicted_states = jax.vmap(model, in_axes=(None, None, 0, 0, 0))(
                None,
                timestamps,
                lags_timestamps,
                encoder_inputs,
                actions_coeffs
            )
            predicted_states = predicted_states.reshape((batch_size * self.prediction_horizon), self.state_size)
            states = future_states.reshape((batch_size * self.prediction_horizon), self.state_size)

            return predicted_states, states

        def validation_step(model, validation_data, batch_size):

            predictions = []
            targets = []
            for step, (val_batch,) in enumerate(self._dataloader_test((validation_data,), batch_size, key=None)):
                val_batch = self.process_batch(val_batch, self.validation_process_key)
                predicted_states, states = make_batch_predictions(
                    model=model,
                    batch=val_batch,
                )
                predictions.append(predicted_states)
                targets.append(states)
                if states.shape[0] == self.prediction_horizon * batch_size:
                    # the last batch of data might be incomplete.
                    predictions.append(predicted_states)
                    targets.append(states)

            total_length = int(len(predictions) * predictions[0].shape[0])
            predictions = jnp.array(predictions).reshape((total_length, 2))
            targets = jnp.array(targets).reshape((total_length, 2))

            if self.state_size == 2:
                power_targets = self.normalizer.denormalize_hvac_power(targets[:, 0])
                power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(predictions[:, 0]), 0)
                mae_power = jnp.mean(jnp.abs((power_targets - power_preds)))
                temperature_targets = self.normalizer.denormalize_temperature(targets[:, 1])
                temperature_preds = self.normalizer.denormalize_temperature(predictions[:, 1])
                mae_temperature = jnp.mean(jnp.abs((temperature_targets - temperature_preds)))
            elif self.model_output == 'power':
                power_targets = self.normalizer.denormalize_hvac_power(targets[:, 0])
                power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(predictions[:, 0]), 0)
                mae_power = jnp.mean(jnp.abs((power_targets - power_preds)))
                temperature_targets = self.normalizer.denormalize_temperature(jnp.zeros(targets[:, 0].shape))
                temperature_preds = self.normalizer.denormalize_temperature(jnp.zeros(targets[:, 0].shape))
                mae_temperature = jnp.mean(jnp.abs((temperature_targets - temperature_preds)))
            else:
                temperature_targets = self.normalizer.denormalize_temperature(targets[:, 1])
                temperature_preds = self.normalizer.denormalize_temperature(predictions[:, 1])
                mae_temperature = jnp.mean(jnp.abs((temperature_targets - temperature_preds)))
                power_targets = self.normalizer.denormalize_hvac_power(jnp.zeros(targets[:, 0].shape))
                power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(jnp.zeros(targets[:, 0].shape)), 0)
                mae_power = jnp.mean(jnp.abs((power_targets - power_preds)))

            return mae_power, mae_temperature

        @eqx.filter_jit
        def loss_fn(model, timestamps, lags_timestamps, inputs, future_states, actions_coeffs):
            y_pred = jax.vmap(model, in_axes=(None, None, 0, 0, 0))(
                None,
                timestamps,
                lags_timestamps,
                inputs,
                actions_coeffs
            )
            if self.state_size == 2:
                power_loss = jnp.mean((future_states[:, :, 0] - y_pred[:, :, 0]) ** 2)
                temperature_loss = jnp.mean((future_states[:, :, 1] - y_pred[:, :, 1]) ** 2)
                loss = power_loss + loss_temperature_weight * temperature_loss
            else:
                loss = jnp.mean((future_states[:, :, 0] - y_pred[:, :, 0]) ** 2)
            return loss

        grad_loss = eqx.filter_value_and_grad(loss_fn)

        @eqx.filter_jit
        def training_step(model, batch, opt_state):
            lags_timestamps, encoder_inputs, timestamps, actions, future_states = batch
            actions_coeffs = jax.vmap(diffrax.backward_hermite_coefficients, in_axes=(None, 0))(
                timestamps,
                actions
            )
            loss, grads = grad_loss(
                model,
                timestamps,
                lags_timestamps,
                encoder_inputs,
                future_states,
                actions_coeffs
            )
            updates, new_opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return new_opt_state, model, loss

        print(f'------ training for {training_steps} steps  ------')

        model = self.model
        training_logs = {'n_steps': [], 'power': [], 'temperature': [], 'mean_train_loss': []}
        optimizer = optax.adabelief(self.learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        losses = []
        for step, (minibatch,) in zip(
                range(training_steps),
                self._dataloader_training((training_data,), training_batch_size, key=self.loader_key)
        ):
            subkey, self.model_key = jrandom.split(self.model_key)
            minibatch = self.process_batch(minibatch, key=subkey)
            if not (minibatch.encoder_inputs is None or minibatch.future_states is None):
                opt_state, model, loss = training_step(
                    model,
                    minibatch,
                    opt_state,
                )
                losses.append(loss)
            else:
                loss = 0.0

            if (validation_data is not None) and ((step % validation_every) == 0 or step == training_steps - 1):
                mae_power, mae_temperature = validation_step(
                    model,
                    validation_data,
                    validation_batch_size,
                )
                training_logs['power'].append(mae_power)
                training_logs['temperature'].append(mae_temperature)
                training_logs['n_steps'].append(step)
                training_logs['mean_train_loss'].append(np.mean(losses))
                print(
                    f"Step: {step}, mean loss: {np.round(np.mean(loss), 8)}, "
                )
                print(
                    f'MAE validation : power {np.round(mae_power, 4)} kW, '
                    f'temperature {np.round(mae_temperature, 4)} degree C'
                )

                # early stoping on power
                if early_stopping and (len(training_logs['power']) > early_stop_lag + 1) and (
                        training_logs['power'][-1] > max(training_logs['power'][-early_stop_lag - 1:-1])):
                    break

        self.model = model
        self._training_logs = training_logs

    def save_model(self, path, final: bool = True):
        if final:
            if not os.path.exists(path):
                os.mkdir(path)
            eqx.tree_serialise_leaves(path + '/observation_encoder.eqx', self.model.obs_encoder)
            eqx.tree_serialise_leaves(path + '/initial_encoder.eqx', self.model.initial_encoder)
            eqx.tree_serialise_leaves(path + '/vector_field.eqx', self.model.vf)
            eqx.tree_serialise_leaves(path + '/decoder.eqx', self.model.decoder)

    def run_model(
        self,
        params,
        timestamps: jnp.array,
        lags_timestamps: jnp.array,
        encoder_inputs: jnp.array,
        actions: jnp.array,
        key=None
    ) -> jnp.array:
        """version of run model that uses jited_model call"""

        actions_coeffs = diffrax.backward_hermite_coefficients(
            timestamps,
            actions
        )
        predicted_states = self.jited_model(
            timestamps,
            lags_timestamps,
            encoder_inputs,
            actions_coeffs
        )

        return predicted_states
