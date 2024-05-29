import numpy as np
import jax.numpy as jnp
from typing import Dict, List


class Normalizer:

    def __init__(self, mean_constants: List[float], std_constants: List[float], indexes: Dict):
        self.means = mean_constants
        self.stds = std_constants
        self.indexes = indexes

    def normalize_hvac_power(self, power):
        mean_power = self.means[self.indexes['hvac_power']]
        std_power = self.stds[self.indexes['hvac_power']]
        return (power - mean_power) / std_power

    def denormalize_hvac_power(self, power):
        mean_power = self.means[self.indexes['hvac_power']]
        std_power = self.stds[self.indexes['hvac_power']]
        return power * std_power + mean_power

    def normalize_temperature(self, temperature):
        mean_temperature = self.means[self.indexes['indoor_temperature']]
        std_temperature = self.stds[self.indexes['indoor_temperature']]
        return (temperature - mean_temperature) / std_temperature

    def denormalize_temperature(self, temperature):
        mean_temperature = self.means[self.indexes['indoor_temperature']]
        std_temperature = self.stds[self.indexes['indoor_temperature']]
        return temperature * std_temperature + mean_temperature

    def normalize_setpoint_change(self, action):
        mean_action = self.means[self.indexes['setpoint_change']]
        std_action = self.stds[self.indexes['setpoint_change']]
        return (action - mean_action) / std_action

    def denormalize_setpoint_change(self, action):
        mean_action = self.means[self.indexes['setpoint_change']]
        std_action = self.stds[self.indexes['setpoint_change']]
        return action * std_action + mean_action

    @staticmethod
    def normalize_lags_timestamps(timestamps: jnp.array) -> jnp.array:
        return (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())

    def normalize_weather(self, weather: np.array):
        """recieve the weather forcast of shape (horizon, n parameters in weather)
        columns are outdoor_temp, humidity, solar rad, day of year, day of week, hour of day"""

        weather = weather.copy()  # copy weather to change it in ray actors

        # Take weather of zone 0, it is the same for every zone
        mean_outdoor_temperature = self.means[self.indexes['outdoor_temperature']]
        std_outdoor_temperature = self.stds[self.indexes['outdoor_temperature']]
        mean_humidity = self.means[self.indexes['humidity']]
        std_humidity = self.stds[self.indexes['humidity']]
        mean_solar_rad = self.means[self.indexes['beam_solar_rad']]
        std_solar_rad = self.stds[self.indexes['beam_solar_rad']]

        weather[:, 0] = (weather[:, 0] - mean_outdoor_temperature) / std_outdoor_temperature
        weather[:, 1] = (weather[:, 1] - mean_humidity) / std_humidity
        weather[:, 2] = (weather[:, 2] - mean_solar_rad) / std_solar_rad

        day_of_week = weather[:, 3]
        hour_of_day = weather[:, 4]
        weather = weather[:, :3]

        norm_calendar_info = np.sin(2 * np.pi * day_of_week/6.0)
        norm_calendar_info = np.vstack((norm_calendar_info, np.cos(2 * np.pi * day_of_week / 6.0)))
        norm_calendar_info = np.vstack((norm_calendar_info, np.sin(2 * np.pi * hour_of_day / 24.0)))
        norm_calendar_info = np.vstack((norm_calendar_info, np.cos(2 * np.pi * hour_of_day / 24.0)))
        norm_weather_and_calendar = np.hstack((weather, np.transpose(norm_calendar_info)))

        return norm_weather_and_calendar

    def denormalize_predictions(self, preds):
        stds = jnp.array([self.stds[self.indexes['hvac_power']], self.stds[self.indexes['indoor_temperature']]])
        means = jnp.array([self.means[self.indexes['hvac_power']], self.means[self.indexes['indoor_temperature']]])

        denorm_pred = preds * stds + means
        return denorm_pred

    def normalize_observations(self, observations: np.array):
        """
        :param observations: observations of shape (lags, 2)
        The order must be power, temperature
        :param zone_id:
        :return:
        """

        norm_observations = np.zeros(observations.shape)
        norm_observations[:, 0] = self.normalize_hvac_power(observations[:, 0])
        norm_observations[:, 1] = self.normalize_temperature(observations[:, 1])
        return norm_observations
