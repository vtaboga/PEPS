import os
import sys
import gin
import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import List, Dict
from energyplus_simulator.energyplus_wrapper import EPlusWrapper
from common.utils import modify_idf_file


@gin.configurable
class System(EPlusWrapper):
    """
    Abstract class to run a simulation
    """

    def __init__(
        self,
        simulation_id: int,
        building_type: str,
        heating_only: bool,
        weather_file: str,
        zones_number: int,
        zones_temperature: List[float],
        controlled_zones_id: List[int],
        observation_lags: int,
        prediction_horizon: int,
        energyplus_timesteps_in_hour: int,
        warmup_iterations: int,
        simulation_start_month: int,
        simulation_start_day: int,
        simulation_end_month: int,
        simulation_end_day: int,
    ):
        """
        :param simulation_id: id of the simulation, used to identify the results
        :param building_type: name of the building used for the simulation.
            Must correspond to an idf file in energyplus_simulator/buildings_blueprint/
        :param weather_file: name of the weather file used for the simulation. Files must be in .epw format.
            Must correspond to an idf file in energyplus_simulator/weather/
        :param zones_number: number of zones in the building
        :param zones_temperature: initial temperature set-point of each zone in the building
        :param controlled_zones_id: id of the zones on which to apply a control
        :param observation_lags: number of previous EnergyPlus timesteps to look at when observing the system
        :param prediction_horizon: number of timesteps in the prediction horizon
        :param energyplus_timesteps_in_hour: number of timesteps per hour in EnergyPlus
        :param warmup_iterations: number of simulation warmup iteration (might change from one building to the other)
        :param simulation_start_month: simulation start month number (January = 1).
            The start day is included in the simulation.
        :param simulation_start_day: simulation start day of the month
        :param simulation_end_month: simulation end month number (January = 1).
            The end date is included in the simulation.
        :param simulation_end_day: simulation end day of the month
        """
        self._results_directory = f'common/results/{building_type}/simulation_{simulation_id}'
        self.create_result_folder(self._results_directory)
        self.horizon = prediction_horizon
        self.obs_lags = observation_lags
        self.zones_predictions = []
        self.controlled_zones_id = controlled_zones_id
        self.new_setpoints = np.zeros(len(controlled_zones_id))
        self.iteration_counter = 0
        self.energyplus_timesteps_in_hour = energyplus_timesteps_in_hour
        self.energyplus_timestep_duration = int(60 / energyplus_timesteps_in_hour)  # time step in minutes
        self.building_type = building_type
        self.simulation_id = simulation_id
        self.logs = None
        self.controllers = None

        self.temp_building_path = modify_idf_file(
            file_path=f'./energyplus_simulator/buildings_blueprint/{building_type}.idf',
            simulation_timestep=energyplus_timesteps_in_hour,
            start_day=simulation_start_day,
            start_month=simulation_start_month,
            end_month=simulation_end_month,
            end_day=simulation_end_day
        )

        EPlusWrapper.__init__(
            self,
            building_type=building_type,
            heating_only=heating_only,
            results_path=self._results_directory,
            weather_path='./energyplus_simulator/weather/' + weather_file,
            building_path=self.temp_building_path,
            zones_number=zones_number,
            zones_temperature=zones_temperature,
            energyplus_timesteps_in_hour=energyplus_timesteps_in_hour,
            warmup_iterations=warmup_iterations
        )

    @abstractmethod
    def run(self) -> None:
        """
        :return: None

        Execute run_simulation with the callbacks specified in the child class
        Additional operations like saving results and configs may be added here
        """
        pass

    def run_simulation(self) -> None:
        """
        :return: None

        Run an EnergyPlus simulation with the building and weather files specified as class parameters
        Callbacks may be overwritten in child classes to run different types of simulation.
        """

        state = self.building.api.state_manager.new_state()
        self.building.api.runtime.clear_callbacks()
        self.building.api.runtime.callback_after_new_environment_warmup_complete(
            state=state,
            f=self.time_step_handler_end_warmup
        )
        self.building.api.runtime.callback_begin_system_timestep_before_predictor(
            state=state,
            f=self.time_step_handler_begin
        )
        self.building.api.runtime.callback_end_zone_timestep_after_zone_reporting(
            state=state,
            f=self.time_step_handler_end
        )

        r = self.building.api.runtime.run_energyplus(
            state,
            command_line_args=[
                '-d',
                self._results_path,
                '-w',
                self._weather_path,
                self._bldg_path]
        )
        if r != 0:
            print("EnergyPlus Failed!")
            sys.exit(1)

    def read_consumption_schedule(self, iteration_counter: int) -> np.array:
        """
        :param iteration_counter: number of EnergyPlus iterations since the beginning of the simulation
        :return: Power targets for the next horizon.

        Parse the consumption schedule and return the consumption targets for the next horizon
        The output is an array of shape (horizon, n_zones) to fit with usual shape,
        but only the power targets of the controlled zones specified in the simulation configuration will be used.
        """

        current_date = self.building.send_current_datetime(iteration_counter)
        next_values = pd.date_range(
            start=current_date,
            periods=self.horizon,
            freq=f'{int(self.building.env_timestep_duration)}s'
        )

        consumption_targets = self.consumption_schedule[
            (self.consumption_schedule['date'] >= next_values[0]) & (
                    self.consumption_schedule['date'] <= next_values[-1])]['power'].to_numpy()

        return consumption_targets

    def send_time_to_dr_event(self, iteration_counter: int):
        current_date = self.building.send_current_datetime(iteration_counter)
        # Find hours of dr events
        dr_times = self.consumption_schedule[
            (self.consumption_schedule['date'].dt.month == current_date.month) &
            (self.consumption_schedule['date'].dt.day == current_date.day) &
            (self.consumption_schedule['power'] != 0)]['date'].to_list()

        if not dr_times:
            return None, None

        start_hour, end_hour = dr_times[0].hour, dr_times[-1].hour + 1

        time_to_start_dr_event = (start_hour - current_date.hour) * self.energyplus_timesteps_in_hour
        time_to_start_dr_event -= int(current_date.minute / 60 * self.energyplus_timesteps_in_hour)
        time_to_end_dr_event = (end_hour - current_date.hour) * self.energyplus_timesteps_in_hour
        time_to_end_dr_event -= int(current_date.minute / 60 * self.energyplus_timesteps_in_hour)

        return time_to_start_dr_event, time_to_end_dr_event

    @abstractmethod
    def time_step_handler_end_warmup(self, state) -> None:
        pass

    @abstractmethod
    def time_step_handler_begin(self, state) -> None:
        pass

    @abstractmethod
    def time_step_handler_end(self, state) -> None:
        pass

    @staticmethod
    def create_result_folder(results_path: str) -> None:
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        if not os.path.exists(results_path + '/data'):
            os.makedirs(results_path + '/data')

