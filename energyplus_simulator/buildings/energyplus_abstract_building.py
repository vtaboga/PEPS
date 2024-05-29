import numpy as np
import pickle
import json
import sys
import datetime
import os

with open(f'common/configurations/path_for_energyplus.json', 'r') as f:
    energyplus_path = json.load(f)
sys.path.insert(0, energyplus_path['path'])

from energyplus_simulator.utils import save_path_handler
from typing import Union
from pyenergyplus.api import EnergyPlusAPI
from abc import ABC, abstractmethod
from typing import List

class EnergyPlusAbstractBuilding(ABC):
    """
    Abstract building class containing the necessary attributes and methods to interact with an EnergyPlus simulation
    """
    api: EnergyPlusAPI
    env_timestep_duration: float
    num_timesteps_in_hour: int
    zones_number: int
    zones_temperature: list
    zones_hvac_power: list
    heating_only: bool
    heating_index: int
    cooling_index: int
    weather: list
    outdoor_temperature: float
    humidity: float
    beam_solar_rad: float
    diffuse_solar: float
    horizontal_solar_ir: float
    wind_speed: float
    wind_direction: float
    day_of_year: int
    day_of_week: int
    hour_of_day: int
    current_time: float
    has_handle: bool
    _total_power_sensor: int
    _internal_load_power_sensor: int
    _outdoor_temp_sensor: int
    _wind_speed_sensor: int
    _wind_direction_sensor: int
    _zones_temperature_sensor: list
    _zones_setpoint_sensor: list
    zones_setpoint_actuator: list
    zones_setpoint_changes: np.array
    zones_temperature_setpoint: np.array
    maximum_memory_length: Union[int, None]
    memory: dict

    def __init__(self, energyplus_timesteps_in_hour: int, heating_only: bool):

        self.api = EnergyPlusAPI()

        # building specifics
        self.heating_only = heating_only
        self.zones_number = 0
        # simulation
        self.num_timesteps_in_hour = energyplus_timesteps_in_hour
        self.env_timestep_duration = 1 / energyplus_timesteps_in_hour * 3600  # in seconds
        self.zone_timestep_duration = None
        self.system_timestep_duration = None
        # powers and temperatures
        self.zones_temperature = []
        self.zones_hvac_power = []
        # weather
        self.weather = []
        self.outdoor_temperature = 0.0
        self.humidity = 0.0
        self.beam_solar_rad = 0.0
        self.diffuse_solar = 0.0
        self.horizontal_solar_ir = 0.0
        self.wind_speed = 0.0
        self.wind_direction = 0.0
        # calendar
        self.day_of_year = 0
        self.day_of_week = 0
        self.hour_of_day = 0
        self.current_time = 0.0
        # sensors
        self.has_handles = False
        self._total_power_sensor = 0
        self._internal_load_power_sensor = 0
        self._outdoor_temp_sensor = 0
        self._wind_speed_sensor = 0
        self._wind_direction_sensor = 0
        self._zones_temperature_sensor = []
        self._zones_heating_setpoint_sensor = []
        self._zones_cooling_setpoint_sensor = []
        self.zones_heating_setpoint_actuator = []
        self.zones_cooling_setpoint_actuator = []
        # actuators
        self.heating_index = 0
        self.cooling_index = 1
        self.zones_setpoint_changes = np.array([])  # size (n_zones, 2), contain heating and cooling setpoints
        self.zones_temperature_setpoint = np.array([])  # size (n_zones, 2), contain heating and cooling setpoints
        # memory
        self.maximum_memory_length = None
        self.memory = {}
        self.reset_memory()

    @abstractmethod
    def observe_hvac_power(self, state) -> None:
        """
        :param state: state of the EnergyPlus simulation returned by the api state_manager
        :return: None

        Update the self.zones_hvac_power attribute

        Read the different meter and variable of EnergyPlus.
        Update the power and energy values of the building with the values of the current timestep.
        This method should be used at the end of a timestep
        !! The variable values depend on a simulation sub-timestep duration. It may lead to unexpected results.
           It is better to work with meter values when possible. !!
        """
        pass

    def observe_temperature(self, state) -> None:
        """
        :param state: state of the EnergyPlus simulation returned by the api state_manager
        :return: None

        Read the different meter and variable of EnergyPlus.
        Update the zones temperature building with the values of the current timestep.
        """

        self.zones_temperature = [
            self.api.exchange.get_variable_value(state, sensor) for sensor in self._zones_temperature_sensor
        ]

    def change_temperature_setpoints(self, state, zone_id: int, setpoint_changes: np.array = None) -> None:

        changes = setpoint_changes if setpoint_changes is not None else self.zones_setpoint_changes
        # change heating setpoint
        self.api.exchange.set_actuator_value(
            state=state,
            actuator_handle=self.zones_heating_setpoint_actuator[zone_id],
            actuator_value=self.zones_temperature_setpoint[zone_id, self.heating_index] + changes[
                zone_id, self.heating_index]
        )
        if not self.heating_only:
            # change cooling setpoint
            self.api.exchange.set_actuator_value(
                state=state,
                actuator_handle=self.zones_cooling_setpoint_actuator[zone_id],
                actuator_value=self.zones_temperature_setpoint[zone_id, self.cooling_index] + changes[
                    zone_id, self.cooling_index]
            )

    def update_setpoint_changes(self, setpoints_changes: np.array, controlled_zones_id: List[int], heating: bool) -> None:
        """
        :param setpoints_changes: shape (number of control zones, planning horizon)
        :param controlled_zones_id: list of controlled zones
        :param heating: wheter to change the heating of cooling setpoint
        :return: None

        Update the zones setpoint changes with the next planned setpoint for each controlled zone
        """

        idx = self.heating_index if heating else self.cooling_index
        self.zones_setpoint_changes[controlled_zones_id, idx] = setpoints_changes[: ,0]

    def observe_weather(self, state) -> None:
        """
        :param state: state of the EnergyPlus simulation returned by the api state_manager
        :return: None

        Update the time and weather.
        The order of the weather list must always be
        [outdoor_temperature, humidity, beam_solar_rad, horizontal_solar_ir, wind_speed, wind_direction]
        """

        # OBSERVE TIME
        timestep = self.api.exchange.zone_time_step_number(state)
        self.day_of_year = self.api.exchange.day_of_year(state)
        self.day_of_week = self.api.exchange.day_of_week(state)
        self.hour_of_day = self.api.exchange.hour(state)
        self.current_time = self.api.exchange.current_time(state)

        # OBSERVE WEATHER
        self.outdoor_temperature = self.api.exchange.get_variable_value(state, self._outdoor_temp_sensor)
        self.beam_solar_rad = self.api.exchange.today_weather_beam_solar_at_time(
            state=state,
            hour=self.hour_of_day,
            time_step_number=timestep
        )
        self.horizontal_solar_ir = self.api.exchange.today_weather_horizontal_ir_at_time(
            state=state,
            hour=self.hour_of_day,
            time_step_number=timestep
        )
        self.diffuse_solar = self.api.exchange.today_weather_diffuse_solar_at_time(
            state=state,
            hour=self.hour_of_day,
            time_step_number=timestep
        )
        self.humidity = self.api.exchange.today_weather_outdoor_relative_humidity_at_time(
            state=state,
            hour=self.hour_of_day,
            time_step_number=timestep
        )
        self.wind_direction = self.api.exchange.get_variable_value(state, self._wind_direction_sensor)
        self.wind_speed = self.api.exchange.get_variable_value(state, self._wind_speed_sensor)
        self.weather = [
            self.outdoor_temperature,
            self.humidity,
            self.beam_solar_rad,
            self.horizontal_solar_ir,
            self.wind_speed,
            self.wind_direction
        ]

    def get_zone_observation(self, zone_id: int, n_lags: int) -> np.array:

        """
        :param zone_id: id of the zone to fetch the observation
        :param n_lags: number of lags of observation to add to the observation.
               if there is less than n_lags of observation in memory,
               the last observation is repeated to unsure shape consistency.
        :return: an array containing the past n_lags observations

        An observation is composed of the power and the temperature.
        Note that if this method is used at the beginning of a time step t, the observations will be
        obs = [obs(t-n_lags), ..., obs(t-2), obs(t-1)].
        """

        energy_observation = np.vstack(self.memory[f'hvac_power_zone{zone_id}'][-n_lags:]).flatten()
        temperature_observation = self.memory[f'temperature_zone{zone_id}'][-n_lags:]
        observation = np.array([energy_observation, temperature_observation])
        n_missing_lags = n_lags - np.shape(observation)[1]
        if n_missing_lags > 0:
            # repeat the last observation to fill the array with n_lags values
            observation = np.hstack((observation, np.tile(observation[:, [-1]], n_missing_lags)))

        return observation

    def observe_zones(self, n_lags: int) -> np.array:
        """
        :param n_lags: number of lags of observation to add to the observation.
                For each zone, if there is less than n_lags of observation in memory,
                the last observation is repeated to unsure shape consistency.
        :return: return array of containing the observations, of shape (n zones, n_lags, obs_dim)
        """

        zones_observation = []
        for zone_id in range(self.zones_number):
            zones_observation.append(self.get_zone_observation(zone_id, n_lags=n_lags))

        return np.array(zones_observation).transpose(0, 2, 1)

    def reset_temperature_setpoints_planning(self, horizon: int) -> np.array:
        """
        :param horizon: planning horizon during with the default setpoints will be applied
        :return: an array of shape (horizon, number of zones in the building)
        """
        return np.tile(self.zones_temperature_setpoint, (horizon, 1))

    def _joule_to_kilowatt_per_system_timestep(self, energy: np.array, state: int) -> np.array:
        """
        :param energy: an array of energy consumption in joule for each zone
        :param state: state of the EnergyPlus simulation returned by the api state_manager
        :return: an array of the same shape as energy,
            containing the average power consumption during the timestep in kiloWatt

        From energy meter (or variable) reading, compute the average power consumption in kW for the current timestep
        !! careful, the system time step may differ from the duration of the timestep the energy was read on.
        Refer to the EnergyPlus documentation for more details. !!
        """
        power = energy / (self.api.exchange.system_time_step(state) * 3600) / 1000

        return power

    def _joule_to_kilowatt_per_simulation_timestep(self, energy: np.array) -> np.array:
        """
        :param energy: an array of energy consumption in joule for each zone
        :param state: state of the EnergyPlus simulation returned by the api state_manager
        :return: an array of the same shape as energy,
            containing the average power consumption during the timestep in kiloWatt

        From energy meter (or variable) reading, compute the average power consumption in kW for the current timestep
        !! careful, the system time step may differ from the duration of the timestep the energy was read on.
        Refer to the EnergyPlus documentation for more details. !!
        """
        power = energy / self.env_timestep_duration / 1000

        return power

    def send_actions_lags(self, n_lags: int, heating: bool = True) -> np.array:

        action_lags = []
        if heating:
            for zid in range(self.zones_number):
                action_lags.append(self.memory[f'heating_setpoint_change_zone{zid}'][-n_lags:].copy())
        else:
            for zid in range(self.zones_number):
                action_lags.append(self.memory[f'cooling_setpoint_change_zone{zid}'][-n_lags:].copy())
        action_lags = np.vstack(action_lags)

        return action_lags

    def send_weather_lags(self, n_lags: int) -> np.array:

        weather_keys = ['outdoor_temperature', 'humidity', 'beam_solar_rad', 'day_of_week', 'hour_of_day']
        weather_lags = []
        for key in weather_keys:
            weather_lags.append(self.memory[key][-n_lags:].copy())
        weather_lags = np.transpose(np.vstack(weather_lags))

        return weather_lags

    def send_actions_forecast(
        self,
        horizon: int
    ):

        actions_forecast = np.array([self.zones_setpoint_changes for _ in range(horizon)])
        actions_forecast = np.vstack(actions_forecast)
        return actions_forecast

    def send_weather_forecast(
        self,
        state,
        horizon: int,
        iteration_counter: int,
        add_noise: bool = False
    ) -> (np.array, np.array):
        """forecast of shape :
        set points, i.e. actions (horizon, n zones)
        weather (horizon, n parameters in weather)
        from timestep t (current) to t+h (h values)"""

        day = self.day_of_year
        day_of_week = self.day_of_week
        hour = self.hour_of_day
        # current_minute = self._api.exchange.minutes(state)
        env_timestep_duration_minute = int(self.env_timestep_duration / 60)
        current_minute = (iteration_counter % self.num_timesteps_in_hour) * env_timestep_duration_minute

        # careful with the order
        weather_forecast = [[
            self.outdoor_temperature,
            self.humidity,
            self.beam_solar_rad,
            self.day_of_week,
            self.hour_of_day
        ]]
        # update time
        current_minute += env_timestep_duration_minute
        if current_minute >= 60:
            hour += 1
            current_minute = current_minute % 60
            if hour >= 24:
                day += 1
                day_of_week += 1
                hour = 0

        for h in range(1, horizon):
            timestep = (current_minute // env_timestep_duration_minute) % self.num_timesteps_in_hour + 1
            if day != self.day_of_year:
                outdoor_temperature = self.api.exchange.tomorrow_weather_outdoor_dry_bulb_at_time(
                    state=state,
                    hour=hour,
                    time_step_number=timestep
                )
                beam_solar_rad = self.api.exchange.tomorrow_weather_beam_solar_at_time(
                    state=state,
                    hour=hour,
                    time_step_number=timestep
                )
                humidity = self.api.exchange.tomorrow_weather_outdoor_relative_humidity_at_time(
                    state=state,
                    hour=hour,
                    time_step_number=timestep
                )
            else:
                outdoor_temperature = self.api.exchange.today_weather_outdoor_dry_bulb_at_time(
                    state=state,
                    hour=hour,
                    time_step_number=timestep
                )
                beam_solar_rad = self.api.exchange.today_weather_beam_solar_at_time(
                    state=state,
                    hour=hour,
                    time_step_number=timestep
                )
                humidity = self.api.exchange.today_weather_outdoor_relative_humidity_at_time(
                    state=state,
                    hour=hour,
                    time_step_number=timestep
                )

            if add_noise:
                noisy_outdoor_temperature = outdoor_temperature + np.random.normal(1) * 0.5
                noisy_humidity = humidity + np.random.normal(1) * 5
                if beam_solar_rad > 0:
                    noisy_beam_solar = max(0.0, beam_solar_rad + np.random.normal(1) * 10)
                else:
                    noisy_beam_solar = 0

                timestep_dist = [
                    noisy_outdoor_temperature,
                    noisy_humidity,
                    noisy_beam_solar,
                    #  day,
                    day_of_week,
                    hour]
            else:
                timestep_dist = [
                    outdoor_temperature,
                    humidity,
                    beam_solar_rad,
                    #  day,
                    day_of_week,
                    hour
                ]

            weather_forecast.append(timestep_dist)
            # update time
            current_minute += env_timestep_duration_minute
            if current_minute >= 60:
                hour += 1
                current_minute = current_minute % 60
                if hour >= 24:
                    day += 1
                    day_of_week += 1
                    hour = 0

        weather_forecast = np.vstack(weather_forecast)

        return weather_forecast

    def send_current_datetime(self, iteration_counter: int) -> datetime.datetime:
        """return the current time in datetime.datetime format"""

        env_timestep_duration_minute = int(self.env_timestep_duration / 60)
        current_minute = (iteration_counter % self.num_timesteps_in_hour) * env_timestep_duration_minute
        current_date = datetime.datetime(2023, 1, 1) + datetime.timedelta(
            days=self.day_of_year - 1,  # date time starts at 1
            hours=self.hour_of_day,
            minutes=current_minute
        )

        return current_date

    @staticmethod
    def _joule_to_kWh(energy: np.array) -> np.array:
        """
        :param energy: energy consumption in joule for each zone
        :return: an array of the same shape as energy,
            containing the average power consumption during the timestep in kWh

        Convert joules to kWh
        """
        return energy / 3600000

    def update_memory(self) -> None:
        """
        :return: None

        Handles the memory of the building.
        Used to store lags of observations and save the results at the end of a simulation.
        The parameter self.maximum_memory_length handles the maximum memory length.
        For a three or four months of simulation, the maximum_memory_length can be let to None
        without having to worry about memory issues.
        """

        def append_memory(memory_key: str, data: Union[list, np.array]) -> None:
            """
            :param memory_key: key of the dictionary corresponding to the data to add
            :param data: list or array to append in memory
            :return: None

            append the data to the memory list for the corresponding key.
            If the memory is full, delete the oldest element.
            """
            self.memory[memory_key].append(data)
            if self.maximum_memory_length is not None and len(self.memory[memory_key]) > self.maximum_memory_length:
                self.memory[memory_key].pop(0)

        append_memory('total_hvac_power', sum(self.zones_hvac_power.copy()))
        append_memory('outdoor_temperature', self.outdoor_temperature)
        append_memory('humidity', self.humidity)
        append_memory('beam_solar_rad', self.beam_solar_rad)
        append_memory('diffuse_solar', self.diffuse_solar)
        append_memory('horizontal_solar_ir', self.horizontal_solar_ir)
        append_memory('wind_speed', self.wind_speed)
        append_memory('wind_direction', self.wind_direction)
        append_memory('day_of_year', self.day_of_year)
        append_memory('day_of_week', self.day_of_week)
        append_memory('hour_of_day', self.hour_of_day)
        append_memory('zone_timestep', self.zone_timestep_duration)
        append_memory('system_timestep', self.system_timestep_duration)
        # Zones specific data
        for zid in range(self.zones_number):
            append_memory(f'hvac_power_zone{zid}', self.zones_hvac_power.copy()[zid])
            append_memory(f'temperature_zone{zid}', self.zones_temperature[zid])
            append_memory(f'heating_setpoint_change_zone{zid}', self.zones_setpoint_changes[zid, self.heating_index])
            append_memory(f'cooling_setpoint_change_zone{zid}', self.zones_setpoint_changes[zid, self.cooling_index])
            append_memory(f'heating_temperature_setpoint_zone{zid}', self.zones_temperature_setpoint[zid, self.heating_index])
            append_memory(f'cooling_temperature_setpoint_zone{zid}', self.zones_temperature_setpoint[zid, self.cooling_index])

    def reset_memory(self) -> None:
        """
        :return: None
        Empty the memory
        """
        self.memory = {
            'total_hvac_power': [],
            'outdoor_temperature': [],
            'humidity': [],
            'beam_solar_rad': [],
            'diffuse_solar': [],
            'horizontal_solar_ir': [],
            'wind_speed': [],
            'wind_direction': [],
            'day_of_year': [],
            'day_of_week': [],
            'hour_of_day': [],
            'zone_timestep': [],
            'system_timestep': []
        }
        for zid in range(self.zones_number):
            self.memory[f'hvac_power_zone{zid}'] = []
            self.memory[f'temperature_zone{zid}'] = []
            self.memory[f'heating_setpoint_change_zone{zid}'] = []
            self.memory[f'cooling_setpoint_change_zone{zid}'] = []
            self.memory[f'heating_temperature_setpoint_zone{zid}'] = []
            self.memory[f'cooling_temperature_setpoint_zone{zid}'] = []

    def save_memory(self, results_path: str, final: bool = True) -> None:
        """
        :param results_path: path to store the memory file
        :param final: If true, save the memory as final. Otherwise, a temp_data file is saved.
                      the temp_data file is deleted when saving the final results
        :return: None
        """
        if final:
            if os.path.exists(results_path + "/data/temp_data.pickle"):
                os.remove(results_path + "/data/temp_data.pickle")
            name = save_path_handler(None, None, '.pickle', results_path)
        else:
            name = results_path + "/data/temp_data.pickle"
        with open(name, 'wb') as handle:
            pickle.dump(self.memory, handle)

    @abstractmethod
    def get_handles(self, state: int) -> None:
        pass

    @abstractmethod
    def _check_handle_values(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_zone_name_from_id(zid: int, complete: bool) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_zone_id_from_name(name: str) -> int:
        pass
