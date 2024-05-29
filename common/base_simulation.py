import gin
import sys
import numpy as np
import os

from typing import List
from common.system import System
from common.utils import save_config_file


@gin.configurable
class BaseSimulation(System):
    
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
        controller_class,
        energyplus_timesteps_in_hour: int,
        warmup_iterations: int,
        simulation_start_month: int,
        simulation_start_day: int,
        simulation_end_month: int,
        simulation_end_day: int,
    ):

        """
        :param controller_class: name of the class of the controller being used in the zones.
            Each zone must have the same controller type.
            This field must correspond to the name of control class in controllers/
        :param simulation_id: id of the simulation, used to identify the results
        :param building_type: name of the building used for the simulation.
            Must correspond to an idf file in energyplus_simulator/buildings_blueprint/
        :param heating_only: is the building equipped with a cooling system
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
        
        System.__init__(
            self,
            simulation_id=simulation_id,
            building_type=building_type,
            heating_only=heating_only,
            weather_file=weather_file,
            zones_number=zones_number,
            zones_temperature=zones_temperature,
            controlled_zones_id=controlled_zones_id,
            observation_lags=observation_lags,
            prediction_horizon=prediction_horizon,
            energyplus_timesteps_in_hour=energyplus_timesteps_in_hour,
            warmup_iterations=warmup_iterations,
            simulation_start_month=simulation_start_month,
            simulation_start_day=simulation_start_day,
            simulation_end_month=simulation_end_month,
            simulation_end_day=simulation_end_day,
        )

        # Instantiate random controllers
        self.controllers = {}
        for zone_id in self.controlled_zones_id:
            self.controllers[f'{zone_id}'] = controller_class(num_timesteps_in_hour=energyplus_timesteps_in_hour)

    def run(self) -> None:
        self.run_simulation()
        self.building.save_memory(self._results_path)
        config_str = gin.operative_config_str()
        save_config_file(config_str, self._results_directory)
        os.remove(self.temp_building_path)

    def time_step_handler_end_warmup(self, state) -> None:
        """
        This method is executed after each warm up iteration
        it checks is the simulation has done its total number of warmup iterations (depending on the building)
        """

        self.building.reset_memory()
        self.iteration_counter = 0
        self._warm_up_count += 1
        self.num_timesteps_in_hour = self.building.api.exchange.num_time_steps_in_hour(state)
        if self._warm_up_count == self._total_warmup_iterations:
            self._done_warm_up = True
            print('warmup count ', self._warm_up_count)
            print('warmup done')
            print('end of the warm up : the memory has been reset')

    def time_step_handler_begin(self, state) -> None:
        """This method is executed at the beginning of each EnergyPlus iteration"""

        sys.stdout.flush()
        if not self.building.has_handles:
            self.building.get_handles(state)

        if self._done_warm_up:
            current_date = self.building.send_current_datetime(self.iteration_counter)
            setpoint_changes = np.zeros((self.zones_number, 2))
            # Draw random setpoint changes using the random controller of each zone
            for i in self.controlled_zones_id:
                self.controllers[f'{i}'].random_setpoint_changes(current_date)
                setpoint_changes[i] = self.controllers[f'{i}'].get_current_setpoint_change()
            # Store the new setpoints in the building class to update the memory
            self.building.zones_setpoint_changes = setpoint_changes
            # Send the new setpoints to EnergyPlus
            for zid in range(self.building.zones_number):
                self.building.change_temperature_setpoints(state=state, zone_id=zid)

            self.building.observe_weather(state)

    def time_step_handler_end(self, state) -> None:
        """This method is executed at the end of each EnergyPlus iteration"""

        sys.stdout.flush()
        if not self.building.has_handles:
            self.building.get_handles(state)

        if self._done_warm_up:
            # Update the temperature and power/energy
            # These values will be the current state of the system at the beginning of the next iteration.
            self.building.observe_temperature(state)
            self.building.observe_hvac_power(state)
            self.building.update_memory()
            self.iteration_counter += 1
