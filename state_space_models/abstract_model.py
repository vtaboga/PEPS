import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import diffrax
import pickle
import pandas as pd
import os

from typing import Dict, List
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from state_space_models.utils.normalizer import Normalizer

ProcessedBatch = namedtuple(
    'ProcessedBatch',
    ['lags_timestamps', 'encoder_inputs', 'timestamps', 'actions', 'future_states']
)


class ABCModel(metaclass=ABCMeta):

    def __init__(
        self,
        seed: int,
        zone_id: int,
        model_type: str,
        observation_size: int,
        state_size: int,
        action_size: int,
        n_lags: int,
        prediction_horizon: int,
        state_indexes: dict,
        include_action_lags: bool,
        include_setpoint_change_difference: bool,
        missing_data_proba: float,
        mean_obs_period: float,
        include_timestep_duration: bool,
        resample_future_states: bool,
        learning_rate: float,
        missing_data_mask: float,
        model_output: str = None,
        mean=None,
        std=None,
    ):
        """
        :param seed: seed used to train the model
        :param state_size: size of the state space
        :param action_size: size of the action space
        :param prediction_horizon: number of timestep to predict
        :param state_indexes: dictionary to map the variable names to their position in the dataset
        :param include_action_lags: whether to include lags of action in the initial state's encoding
        :param include_setpoint_change_difference: whether to include the difference between to consecutive setpoints
        :param missing_data_proba: probability of not observing an entry in the lags of observation
        :param mean_obs_period: Poisson constant defining the mean duration between two observations.
                This parameter is used in the irregularly sampled data setting.
        :param resample_future_states: whether to resample future state to change the timestep duration
                This parameter is useful is a short timestep was used to collect data,
                but you want the predictions to be made on longer time steps.
        :param learning_rate: learning rate used to train the model.
        :param mean: mean values of the training data. Used for normalization
        :param std: std of the training data variables. Used for normalization
        """

        if model_type is not None and model_type != 'stochastic':
            assert model_type in ['discrete', 'continuous']
        self.continuous_model = (model_type == 'continuous')
        self.zone_id = zone_id

        self.prediction_horizon = prediction_horizon
        self.missing_data_proba = missing_data_proba
        self.mean_obs_period = mean_obs_period
        self.resample_future_states = resample_future_states
        self.include_timestep_duration = include_timestep_duration
        self.state_size = state_size
        self.action_size = action_size + 1 if include_setpoint_change_difference else action_size
        self.include_action_lags = include_action_lags
        self.include_setpoint_change_difference = include_setpoint_change_difference
        self.n_lags = n_lags
        self.missing_data_mask = missing_data_mask
        self.observation_size = observation_size + self.action_size if include_action_lags else observation_size
        self.encoder_in_size = self.observation_size + 1 if self.include_timestep_duration else self.observation_size
        self.actions_memory = []
        self._training_logs = {}
        self.learning_rate = learning_rate
        if mean is not None and std is not None:
            self.normalizer = Normalizer(
                mean_constants=mean,
                std_constants=std,
                indexes=state_indexes
            )
        self.indexes = state_indexes
        self.model_output = model_output
        # Adapt the state size to the prediction output specified
        if state_size == 1:
            if self.model_output == 'hvac_power':
                self.state_indexes = [state_indexes['hvac_power']]
            elif self.model_output == 'indoor_temperature':
                self.state_indexes = [state_indexes['indoor_temperature']]
            else:
                raise ValueError(f'Model output {model_output} not defined.')
        elif state_size == 2:
            self.state_indexes = [state_indexes['hvac_power'], state_indexes['indoor_temperature']]
        else:
            raise ValueError(f'State size incorrect. Got {state_size}, should be 1 or 2.')
        self.input_indexes = self.get_inputs_indexes(
            include_action_lags,
            state_indexes,
            include_setpoint_change_difference
        )
        self.timestamps_lag_indexes = self.get_timestamps_lag_indexes(state_indexes)
        self.action_indexes = self.get_action_indexes(
            state_indexes=state_indexes,
            include_setpoint_change_difference=include_setpoint_change_difference
        )
        self.seed = seed
        key = jrandom.PRNGKey(self.seed)
        self.model_key, self.loader_key, self.validation_process_key, self.test_process_key = jrandom.split(key, 4)
        self.model = None
        self.model_parameters = None

    @abstractmethod
    def initialize_model(self) -> Dict:
        """
        :return: Dictionary of the initial weights of the model
        """
        pass

    def load_model(self, model_path: str) -> None:
        """
        :param model_path: path to the saved model weights
        :return: None

        If a saving exist, load the model parameters in the attribute self.model_parameters
        """

        if model_path is not None:
            with open(model_path + f'/model_parameters.pkl', "rb") as f:
                self.model_parameters = pickle.load(f)
        else:
            self.model_parameters = None

    def train_model(
        self,
        training_data,
        validation_data,
        loss_temperature_weight,
        training_batch_size,
        validation_batch_size,
        early_stopping,
        training_steps: int,
        early_stop_lag=2,
        validation_every=500,
        model_path: str = None
    ) -> None:

        """
        :param zone_id: number of the zone the model in trained on
        :param training_data: training dataset. The data should normalized
        :param validation_data: validation dataset. The data should be normalized
        :param loss_temperature_weight: weight on the temperature prediction objective.
        :param training_batch_size: batch size used for training
        :param validation_batch_size: batch size used for validation (may affect the computation duration)
        :param training_steps: number of training steps
        :param early_stop_lag: number of previous validation score to consider for the early stopping
        :param validation_every: number of training steps between two validations
        :param early_stopping: whether to apply early stopping to the training based on the validation score
        :param model_path: path to store the model weights and configuration
        :return: None

        Train a model and save the weights.
        Predictions are made differently for discrete and continuous time models. This method handles both cases and is
        parametrized by the boolean self.continuous_model.
        """

        @jax.jit
        def make_batch_predictions(params, batch):
            # Batch predictions for validation
            lags_timestamps, encoder_inputs, timestamps, actions, future_states = batch
            batch_size = future_states.shape[0]
            if self.continuous_model:
                actions_coeffs = jax.vmap(diffrax.backward_hermite_coefficients, in_axes=(None, 0))(
                    timestamps,
                    actions
                )
                predicted_states = jax.vmap(self.model, in_axes=(None, None, 0, 0, 0))(
                    params,
                    timestamps,
                    lags_timestamps,
                    encoder_inputs,
                    actions_coeffs
                )
            else:
                predicted_states = jax.vmap(self.model.apply, in_axes=(None, None, 0, 0))(
                    params,
                    None,
                    encoder_inputs,
                    actions
                )
            predicted_states = predicted_states.reshape((batch_size * self.prediction_horizon), self.state_size)
            states = future_states.reshape((batch_size * self.prediction_horizon), self.state_size)

            return predicted_states, states

        def validation_step(params, validation_data, batch_size):

            predictions = []
            targets = []
            for step, (val_batch,) in enumerate(self._dataloader_test((validation_data,), batch_size, key=None)):
                val_batch = self.process_batch(val_batch, self.validation_process_key)
                if val_batch.encoder_inputs is None or val_batch.future_states is None:
                    continue
                predicted_states, states = make_batch_predictions(
                    params=params,
                    batch=val_batch,
                )
                if states.shape[0] == self.prediction_horizon * batch_size:
                    # the last batch of data might be incomplete.
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
                temperature_targets = self.normalizer.denormalize_temperature(jnp.zeros(targets[:, 0].shape))
                temperature_preds = self.normalizer.denormalize_temperature(jnp.zeros(targets[:, 0].shape))
                mae_temperature = jnp.mean(jnp.abs((temperature_targets - temperature_preds)))
            else:
                temperature_targets = self.normalizer.denormalize_temperature(targets[:, 0])
                temperature_preds = self.normalizer.denormalize_temperature(predictions[:, 0])
                mae_temperature = jnp.mean(jnp.abs((temperature_targets - temperature_preds)))
                hvac_power_targets = self.normalizer.denormalize_hvac_power(jnp.zeros(targets[:, 0].shape))
                hvac_power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(jnp.zeros(targets[:, 0].shape)), 0)
                mae_hvac_power = jnp.mean(jnp.abs((hvac_power_targets - hvac_power_preds)))

            return mae_hvac_power, mae_temperature

        @jax.jit
        def loss_fn(params, timestamps, lags_timestamps, inputs, future_states, actions):
            if self.continuous_model:
                y_pred = jax.vmap(self.model, in_axes=(None, None, 0, 0, 0))(
                    params,
                    timestamps,
                    lags_timestamps,
                    inputs,
                    actions  # actions coefficients from interpolation
                )
            else:
                y_pred = jax.vmap(self.model.apply, in_axes=(None, None, 0, 0))(
                    params,
                    None,
                    inputs,
                    actions
                )
            if self.state_size == 2:
                hvac_power_loss = jnp.mean((future_states[:, :, 0] - y_pred[:, :, 0]) ** 2)
                temperature_loss = jnp.mean((future_states[:, :, 1] - y_pred[:, :, 1]) ** 2)
                loss = hvac_power_loss + loss_temperature_weight * temperature_loss
            else:
                loss = jnp.mean((future_states[:, :, 0] - y_pred[:, :, 0]) ** 2)
            return loss

        @jax.jit
        def training_step(params, batch, opt_state):
            lags_timestamps, encoder_inputs, timestamps, actions, future_states = batch
            if self.continuous_model:
                actions_coeffs = jax.vmap(diffrax.backward_hermite_coefficients, in_axes=(None, 0))(
                    timestamps,
                    actions
                )
                loss, grads = jax.value_and_grad(loss_fn)(
                    params,
                    timestamps,
                    lags_timestamps,
                    encoder_inputs,
                    future_states,
                    actions_coeffs
                )
            else:
                loss, grads = jax.value_and_grad(loss_fn)(
                    params,
                    timestamps,
                    lags_timestamps,
                    encoder_inputs,
                    future_states,
                    actions
                )
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_opt_state, new_params, loss

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
            subkey, self.model_key = jrandom.split(self.model_key)
            minibatch = self.process_batch(minibatch, key=subkey)
            if not (minibatch.encoder_inputs is None or minibatch.future_states is None):
                opt_state, params, loss = training_step(
                    params,
                    minibatch,
                    opt_state,
                )
                losses.append(loss)
            else:
                loss = 0.0

            if (validation_data is not None) and ((step % validation_every) == 0 or step == training_steps - 1):
                mae_hvac_power, mae_temperature = validation_step(
                    params,
                    validation_data,
                    validation_batch_size,
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
                if self.model_output == 'hvac_power' or self.model_output is None:
                    if early_stopping and (len(training_logs['hvac_power']) > early_stop_lag + 1) and (
                            training_logs['hvac_power'][-1] > max(
                        training_logs['hvac_power'][-early_stop_lag - 1:-1]
                    )):
                        break

        self.model_parameters = params
        self._training_logs = training_logs

    def missing_data(self, encoder_inputs: jnp.array, key: jnp.array) -> jnp.array:
        """
        :param encoder_inputs: encoder inputs in shape (prediction_horizon, action_dim)
        :param key: jax random key generator
        :return: encoder inputs with entries put to zero.
        """

        random_mask = jax.random.uniform(key, encoder_inputs.shape)
        missing_data_mask = jnp.where(random_mask < self.missing_data_proba, 0, 1)
        encoder_inputs = encoder_inputs * missing_data_mask

        return encoder_inputs

    def irregularly_sampled_data(
        self,
        encoder_inputs: jnp.array,
        lags_timestamps: jnp.array,
        key: jnp.array
    ) -> jnp.array:
        """
        :param encoder_inputs: encoder inputs in shape (prediction_horizon, action_dim)
        :param lags_timestamps: timestamps of the observation lags.
        :param key: jax random key generator
        :return: encoder inputs with missing columns, of shape (a, action_dim) with a <= prediction_horizon
        """

        n_data_points = encoder_inputs.shape[0]
        steps_to_next_observation = jax.random.poisson(
            key,
            self.mean_obs_period,
            shape=(n_data_points + 1,)
        ) + 1  # non null steps to the next obs
        irregular_indices = np.cumsum(steps_to_next_observation)
        irregular_indices = irregular_indices[1:] - 1  # The last lag is not necessarily observed.
        irregular_indices = irregular_indices[irregular_indices < n_data_points]
        encoder_inputs = encoder_inputs[irregular_indices, :]
        lags_timestamps = lags_timestamps[irregular_indices]

        return encoder_inputs, lags_timestamps

    def process_batch(self, batch: jnp.array, key: jnp.array = None):
        """
        :param batch: batch of data of shape (batch_size, prediction horizon, -- )
        :param key: jax random key generator
        :return: the lags timestamps, encoder inputs in shape (batch_size, n_lags, input dim)
                 the prediction timestamps, the actions in shape (batch_size, prediction_horizon, action dim)
                 the future states in shape (batch_size, prediction_horizon, state dim)
        """
        batch_size, _, _ = batch.shape
        encoder_inputs = batch[:, 0, self.input_indexes]
        n_observations = encoder_inputs.shape[1] // self.observation_size
        encoder_inputs = encoder_inputs.reshape((batch_size, n_observations, self.observation_size))
        encoder_inputs = jnp.flip(encoder_inputs, axis=1)
        lags_timestamps = jnp.flip(batch[:, 0, self.timestamps_lag_indexes], axis=1)
        lags_timestamps = jax.vmap(self.normalizer.normalize_lags_timestamps)(lags_timestamps)
        if self.mean_obs_period > 0:
            subkey, key = jrandom.split(key)
            # use the same random key for every element of the batch to keep a uniform shape
            encoder_inputs, lags_timestamps = jax.vmap(self.irregularly_sampled_data, in_axes=(0, 0, None))(
                encoder_inputs,
                lags_timestamps,
                subkey
            )
        if self.missing_data_proba > 0:
            subkey, key = jrandom.split(key)
            encoder_inputs = self.missing_data(encoder_inputs, subkey)
        if self.include_timestep_duration:
            timesteps = jax.vmap(self.compute_timesteps)(lags_timestamps)
            encoder_inputs = jnp.concatenate(
                [jnp.expand_dims(timesteps, axis=2), encoder_inputs], axis=-1
            )
        if self.resample_future_states:
            future_states = self.resample(batch[:, :, self.state_indexes])
            actions = self.resample(batch[:, :, self.action_indexes])
        else:
            future_states = batch[:, :, self.state_indexes]
            actions = batch[:, :, self.action_indexes]
        timestamps = jnp.linspace(
            0,
            1,
            self.prediction_horizon
        )

        # handle NaN is data
        actions = jnp.where(jnp.isnan(actions), self.missing_data_mask, actions)
        if jnp.any(jnp.isnan(future_states)):
            future_states = None
        if jnp.any(jnp.isnan(encoder_inputs)):
            future_states = None

        return ProcessedBatch(lags_timestamps, encoder_inputs, timestamps, actions, future_states)

    def make_batch_predictions(self, params: Dict, batch: jnp.array) -> (jnp.array, jnp.array):
        """
        :param params: parameters (weights) of the model
        :param batch: batch of data for the predictions.
        make predictions for an ordered batch of shape (batch_length, trajectory_length, state_size)
        the predictions are made using jax vmap and put back in the shape (batch_length * trajectory_length, sate_size)
        """

        subkey1, subkey2, self.test_process_key = jax.random.split(self.test_process_key, num=3)
        lags_timestamps, encoder_inputs, timestamps, actions, future_states = self.process_batch(batch, subkey1)
        if encoder_inputs is None or future_states is None:
            return None, None
        batch_size = future_states.shape[0]
        predicted_states = jax.vmap(self.run_model, in_axes=(None, None, 0, 0, 0, None))(
            params,
            timestamps,
            lags_timestamps,
            encoder_inputs,
            actions,
            subkey2
        )
        predicted_states = predicted_states.reshape((batch_size * self.prediction_horizon), self.state_size)
        states = future_states.reshape((batch_size * self.prediction_horizon), self.state_size)

        return predicted_states, states

    def test_model(self, batch_size: int, test_data: jnp.array) -> Dict:
        """
        :param batch_size: size of the batch to parallelize the tests
        :param test_data: test dataset (normalized)
        :return: Dictionary containing the state predictions and true values
        """
        predictions = []
        targets = []
        for step, (test_batch,) in enumerate(self._dataloader_test((test_data,), batch_size, key=None)):
            predicted_states, states = self.make_batch_predictions(
                self.model_parameters,
                test_batch
            )
            if predicted_states is None:
                continue
            predictions.append(predicted_states)
            targets.append(states)
            if states.shape[0] == self.prediction_horizon * batch_size:
                # the last batch of data might be incomplete.
                predictions.append(predicted_states)
                targets.append(states)
        total_length = int(len(predictions) * predictions[0].shape[0])
        predictions = jnp.array(predictions).reshape((total_length, self.state_size))
        targets = jnp.array(targets).reshape((total_length, self.state_size))

        # Adapt the results to the specified output for the prediction model
        if self.state_size == 2:
            hvac_power_targets = self.normalizer.denormalize_hvac_power(targets[:, 0])
            hvac_power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(predictions[:, 0]), 0)
            temperature_targets = self.normalizer.denormalize_temperature(targets[:, 1])
            temperature_preds = self.normalizer.denormalize_temperature(predictions[:, 1])
        elif self.model_output == 'hvac_power':
            hvac_power_targets = self.normalizer.denormalize_hvac_power(targets[:, 0])
            hvac_power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(predictions[:, 0]), 0)
            temperature_targets = self.normalizer.denormalize_temperature(jnp.zeros(targets[:, 0].shape))
            temperature_preds = self.normalizer.denormalize_temperature(jnp.zeros(targets[:, 0].shape))
        else:
            temperature_targets = self.normalizer.denormalize_temperature(targets[:, 0])
            temperature_preds = self.normalizer.denormalize_temperature(predictions[:, 0])
            hvac_power_targets = self.normalizer.denormalize_hvac_power(jnp.zeros(targets[:, 0].shape))
            hvac_power_preds = jnp.clip(self.normalizer.denormalize_hvac_power(jnp.zeros(targets[:, 0].shape)), 0)

        results = {
            'hvac_power_targets': hvac_power_targets,
            'hvac_power_predictions': hvac_power_preds,
            'temperature_targets': temperature_targets,
            'temperature_predictions': temperature_preds
        }

        return results

    def make_predictions(self, inputs, actions, energyplus_timestep_duration, prediction_horizon, *args):
        """
        :param inputs: normalized encoder inputs of shape (n_lags, observation dim)
        :param actions:  normalized actions of shape (horizon, action_dim)
        :param energyplus_timestep_duration: time step of the simulation in minutes
        :param prediction_horizon: number of time steps to predict
        :return: denormalized predictions in shape (horizon, n observations). Usually (power, temperature)
        """

        lags_timestamps = jnp.linspace(0, (self.n_lags - 1), self.n_lags) * energyplus_timestep_duration
        min_lag, max_lag = jnp.min(lags_timestamps), jnp.max(lags_timestamps)
        lags_timestamps = (lags_timestamps - min_lag) / (max_lag - min_lag)
        timestamps = jnp.linspace(0, 1, prediction_horizon)
        predictions = self.run_model(self.model_parameters, timestamps, lags_timestamps, inputs, actions)
        predictions = self.normalizer.denormalize_predictions(predictions)

        return predictions

    @abstractmethod
    def run_model(
        self,
        params: Dict,
        timestamps: jnp.array,
        lags_timestamps: jnp.array,
        inputs: jnp.array,
        actions: jnp.array,
        key: jnp.array = None
    ) -> jnp.array:
        """
        :param params: model parameters (weights)
        :param timestamps: prediction timestamps.
        :param lags_timestamps: observations timestamps
        :param inputs: lags of observations
        :param actions: actions (gathers setpoint changes, calendar information and weather forecasts)
        :param key: random key if the model is stochastic
        :return: array of state predictions of the prediction horizon specified by the timestamps variable
        """
        pass

    def save_model(self, path: str, final: bool) -> None:
        """
        :param path: path to the folder to save the model's parameters
        :param zone_id:
        :param final: whether this is the final models parameters. If not, tmp is added at the end of the file's name.
        :return: None
        """
        if final:
            with open(path + f'/model_parameters.pkl', "wb") as f:
                pickle.dump(self.model_parameters, f)
            # delete model saved while training
            if os.path.isfile(path + f'/model_parameters_tmp.pkl'):
                os.remove(path + f'/model_parameters_tmp.pkl')
        else:
            with open(path + f'/model_parameters_tmp.pkl', "wb") as f:
                pickle.dump(self.model_parameters, f)

    def save_training_logs(self, path: str) -> None:
        """
        :param path: path to the folder to save the training logs
        :return: None
        """
        if self._training_logs is not None:
            pd.DataFrame(self._training_logs).to_csv(path + f'/training_logs.csv')
        else:
            raise ValueError(f'Cannot save the training logs, training logs is {self._training_logs}')

    @staticmethod
    def _dataloader_training(arrays: jnp.array, batch_size: int, key: jnp.array):
        """
        :param arrays: processed training data
        :param batch_size:
        :param key: jax random key generator
        :return: load batches of training data.
        """

        dataset_size = arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in arrays)
        indices = jnp.arange(dataset_size)
        if key is None:
            while True:
                start = 0
                end = batch_size
                while end < dataset_size:
                    batch_indices = indices[start:end]
                    yield tuple(array[batch_indices] for array in arrays)
                    start = end
                    end = start + batch_size
        else:
            while True:
                perm = jax.random.permutation(key, indices)
                (key,) = jax.random.split(key, 1)
                start = 0
                end = batch_size
                while end < dataset_size:
                    batch_perm = perm[start:end]
                    yield tuple(array[batch_perm] for array in arrays)
                    start = end
                    end = start + batch_size

    @staticmethod
    def _dataloader_test(arrays: jnp.array, batch_size: int, key: jnp.array):
        """
        :param arrays: processed test data
        :param batch_size:
        :param key: jax random key generator
        :return: load batches of test data.
        """
        dataset_size = arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in arrays)
        indices = jnp.arange(dataset_size)

        if key is None:
            start = 0
            while start < dataset_size:
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                yield tuple(array[batch_indices] for array in arrays)
                start = end
        else:
            perm = jax.random.permutation(key, indices)
            start = 0
            while start < dataset_size:
                end = min(start + batch_size, dataset_size)
                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in arrays)
                start = end

    @staticmethod
    def compute_timesteps(timestamps: jnp.array) -> jnp.array:

        timesteps = jnp.diff(timestamps)
        timesteps = jnp.concatenate([jnp.array([0.]), timesteps])

        return timesteps

    def resample(self, inputs: jnp.array) -> jnp.array:
        """
        :param inputs: either future state or action, shape (batch_size, n_steps, - )
        :return: resample the frequency of the array to match the prediction horizon
        Use this method if the prediction step size is lower than the simulation step size
        """
        n_future_states = inputs.shape[1]
        resampling_frequency = n_future_states // self.prediction_horizon
        indexes = jnp.arange(start=0, stop=n_future_states, step=resampling_frequency)
        return inputs[:, indexes, :]

    @staticmethod
    def get_state_indexes(n_lags_in_data: int, state_size: int = 2, n_lags: int = None):
        """
        n_lags_in_data: n_lags parameter used to parse the data
        n_lags: lags to use of the observation
        return the indexes of the hvac_power and temperature observations the input data
        """
        n_lags = n_lags_in_data if n_lags is None else n_lags
        if state_size != 2:
            raise ValueError(
                f'the state size must be equal to 2 (hvac_power, temperature) to get the observation indexes'
            )
        hvac_power_indexes = [i for i in range(1, n_lags_in_data + 1)]
        temperature_indexes = [i for i in range(n_lags_in_data + 2, 2 * n_lags_in_data + 2)]
        state_indexes = hvac_power_indexes[:n_lags] + temperature_indexes[:n_lags]
        return sorted(state_indexes)

    @staticmethod
    def get_timestamps_lag_indexes(state_indexes):
        indexes = [value for key, value in state_indexes.items() if 'timestamp_lag' in key]
        return sorted(indexes)

    @staticmethod
    def get_inputs_indexes(
        include_action_lags,
        state_indexes,
        include_setpoint_change_difference,
        one_lag: bool = False,
    ):

        if one_lag:
            if include_action_lags:
                if include_setpoint_change_difference:
                    indexes = [value for key, value in state_indexes.items() if
                        key.endswith('lag_1') and 'timestamp' not in key]
                else:
                    indexes = [value for key, value in state_indexes.items() if
                        key.endswith('lag_1') and ('timestamp' not in key and 'delta' not in key)]
            else:
                indexes = [value for key, value in state_indexes.items() if key.endswith('indoor_temperature_lag_1')]
                indexes += [value for key, value in state_indexes.items() if key.endswith('hvac_power_lag_1')]
        else:
            if include_action_lags:
                if include_setpoint_change_difference:
                    indexes = [value for key, value in state_indexes.items() if 'lag' in key and 'timestamp' not in key]
                else:
                    indexes = [value for key, value in state_indexes.items() if
                        'lag' in key and ('timestamp' not in key and 'delta' not in key)]
            else:
                indexes = [value for key, value in state_indexes.items() if 'indoor_temperature_lag' in key]
                indexes += [value for key, value in state_indexes.items() if 'hvac_power_lag' in key]

        return sorted(indexes)

    @staticmethod
    def get_action_indexes(state_indexes, include_setpoint_change_difference: bool = False):

        if include_setpoint_change_difference:
            action = ['outdoor_temperature', 'humidity', 'beam_solar_rad', 'day_of_week_sin', 'day_of_week_cos',
                'hour_of_day_cos', 'hour_of_day_sin', 'setpoint_change', 'delta_setpoint_change']
        else:
            action = ['outdoor_temperature', 'humidity', 'beam_solar_rad', 'day_of_week_sin', 'day_of_week_cos',
                'hour_of_day_cos', 'hour_of_day_sin', 'setpoint_change']

        indexes = [value for key, value in state_indexes.items() if key in action]

        return sorted(indexes)
