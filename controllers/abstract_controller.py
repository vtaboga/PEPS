import numpy as np
import jax.numpy as jnp

from state_space_models.utils.normalizer import Normalizer
from abc import abstractmethod, ABC
from typing import Union, List, Dict


class ABCController(ABC):

    def __init__(self):
        self.prediction_model = None

    @abstractmethod
    def run(self, *args) -> Union[np.ndarray, jnp.ndarray]:
        """return the control actions (i.e. temperature set point changes)"""
        pass

    def reset_controller(self) -> None:
        pass

    def process_input_data(
        self,
        observations: np.array,
        weather_forecast: np.array,
        actions_forecast: Union[None, np.array],
        actions_lags: Union[None, np.array],
        weather_lags: Union[None, np.array]
    ) -> Dict:
        """
        Process input data by normalizing observations and weather forecasts, handling lags, and optionally including actions forecasts.
        Returns a dictionary with 'observation' and 'actions' as keys.
        """

        # Normalize weather and observations
        norm_weather = self.prediction_model.normalizer.normalize_weather(weather_forecast)
        norm_observations = self.prediction_model.normalizer.normalize_observations(observations=observations)
        action_position = self.prediction_model.indexes['setpoint_change'] - norm_observations.shape[1]

        # Handle lags if required
        if self.prediction_model.include_action_lags:
            if weather_lags is None or actions_lags is None:
                raise ValueError('Both weather_lags and actions_lags are required.')
            norm_weather_lags = self.prediction_model.normalizer.normalize_weather(weather_lags)
            # Insert action lags at the specified action position
            norm_action_lags = self.prediction_model.normalizer.normalize_setpoint_change(actions_lags[1:])
            inputs = jnp.append(norm_weather_lags, jnp.expand_dims(norm_action_lags, axis=1), axis=1)
            inputs = jnp.append(norm_observations, inputs, axis=1)
            # Include differences if specified
            if self.prediction_model.include_setpoint_change_difference:
                # setpoint differneces are never normalizer
                delta_action_lags = jnp.diff(actions_lags)
                inputs = jnp.concatenate((inputs, jnp.expand_dims(delta_action_lags, axis=1)), axis=1)

        else:
            inputs = norm_observations

        # Insert action forecasts if available
        if actions_forecast is not None:
            norm_actions = self.prediction_model.normalizer.normalize_setpoint_change(actions_forecast)
            actions = jnp.insert(norm_weather, action_position, norm_actions, axis=1)
            if self.prediction_model.include_setpoint_change_difference:
                delta_actions = jnp.diff(jnp.concatenate((actions_lags[-1:], actions_forecast)))
                actions = jnp.concatenate((actions, jnp.expand_dims(delta_actions, axis=1)), axis=1)
        else:
            actions = norm_weather

        input_data = {'inputs': inputs, 'actions': actions}
        return input_data

    def make_predictions(
        self,
        current_obs: np.array,
        setpoint_changes: np.array,
        weather_forecast: np.array,
        weather_lags: Union[None, np.array],
        action_lags: Union[None, np.array],
        energyplus_timestep_duration: int,
        prediction_horizon: int
    ) -> Dict:
        """
        :param current_obs:
        :param actions_forecast:
        :param weather_forecast:
        :param energyplus_timestep_duration:
        :param horizon:
        :return:
        """
        print("--- ShootingController.make_predictions ---")

        inputs = self.process_input_data(
            observations=current_obs,
            weather_forecast=weather_forecast,
            actions_forecast=setpoint_changes,
            weather_lags=weather_lags,
            actions_lags=action_lags
        )

        print(f"input.keys(): {inputs.keys()}")

        predictions = self.prediction_model.make_predictions(
            inputs=inputs['inputs'],
            actions=inputs['actions'],
            energyplus_timestep_duration=energyplus_timestep_duration,
            prediction_horizon=prediction_horizon
        )
        # unpack results
        power_prediction = predictions[:, 0]
        power_prediction = jnp.where(power_prediction < 0, 0, power_prediction)  # no negative powers
        temperature_prediction = predictions[:, 1]
        predictions = {'power': power_prediction, 'temperature': temperature_prediction}

        return predictions

    def process_power_schedule(
        self,
        observation,
        weather,
        action_lags,
        weather_lags,
        power_schedule: jnp.array,
        energyplus_timestep_duration: int,
        logs_file: str
    ) -> Union[None, jnp.array]:
        """
        :param zone_observation:
        :param weather:
        :param power_schedule:
        :param logs_file:
        :return:
        """

        if jnp.all(power_schedule == 0):
            # No maximum power constraint during the next horizon
            return None
        else:
            setpoints = jnp.zeros(self.prediction_model.prediction_horizon)
            predictions = self.make_predictions(
                current_obs=observation,
                setpoint_changes=setpoints,
                weather_forecast=weather,
                weather_lags=weather_lags,
                action_lags=action_lags,
                energyplus_timestep_duration=energyplus_timestep_duration,
                prediction_horizon=self.prediction_model.prediction_horizon
            )

            power_prediction = predictions['power']
            with open(logs_file, 'a') as f:
                f.write(f"---------------- CHECK IF ACTIONS ARE NEEDED ----------------\n")
                f.write(f'Power prediction {power_prediction} \n')
                f.write(f'Power maximum {power_schedule} \n')

            if jnp.all(power_prediction[power_schedule != 0] < power_schedule[power_schedule != 0]):
                # With no set point change, the max power constraint is satisfied
                return None
            else:
                maximum_power_schedule = jnp.where(power_schedule == 0, jnp.inf, power_schedule)
                return maximum_power_schedule.flatten()
