import sys
import gin
import numpy as np
import pandas as pd
import jax.numpy as jnp

from typing import List
from common.system import System
from common.utils import save_control_memory, get_log_filename
from off_line_computation.utils import load_norm_constants, get_processed_state_indexes


@gin.configurable
class SingleZoneBuildingSimulation(System):

    def __init__(
        self,
        simulation_id: int,
        building_type: str,
        slack_power_constraint: float,
        heating_only: bool,
        heating_period: bool,
        weather_file: str,
        zones_temperature: List[float],
        observation_lags: int,
        prediction_horizon: int,
        controller_class,
        model_path: str,
        consumption_schedule_name: str,
        energyplus_timesteps_in_hour: int,
        warmup_iterations: int,
        simulation_start_month: int,
        simulation_start_day: int,
        simulation_end_month: int,
        simulation_end_day: int,
    ):

        System.__init__(
            self,
            simulation_id=simulation_id,
            building_type=building_type,
            heating_only=heating_only,
            weather_file=weather_file,
            zones_number=1,  # This is a single zone building
            zones_temperature=zones_temperature,
            controlled_zones_id=[0],  # Controlling the building's unique zone
            observation_lags=observation_lags,
            prediction_horizon=prediction_horizon,
            energyplus_timesteps_in_hour=energyplus_timesteps_in_hour,
            warmup_iterations=warmup_iterations,
            simulation_start_month=simulation_start_month,
            simulation_start_day=simulation_start_day,
            simulation_end_month=simulation_end_month,
            simulation_end_day=simulation_end_day,
        )

        try:
            schedule_path = f'common/configurations/consumption_schedule/{consumption_schedule_name}.csv'
            self.consumption_schedule = pd.read_csv(schedule_path)
            self.consumption_schedule['date'] = pd.to_datetime(self.consumption_schedule['date'])
        except FileNotFoundError as e:
            print(f'could not load the consumption schedule, got error {e}')
            raise e

        self.zone_id = 0  # The building has a single zone
        self.heating_period = heating_period
        data_path = f'common/results/{building_type}/simulation_{simulation_id}/processed_data'
        mean, std = load_norm_constants(self.zone_id, data_path)
        state_indexes = get_processed_state_indexes(data_path)
        self.controller = controller_class(
            zone_id=0,  # Single zone building
            mean_constants=mean,
            std_constants=std,
            state_indexes=state_indexes,
            model_path=model_path
        )
        self.n_lags = self.controller.prediction_model.n_lags
        self.slack_power_constraint = slack_power_constraint
        self.control_memory = {
            'power_predictions': [],
            'power_schedule': [],
            'slacked_power_schedule': [],
            'power': [],
            'temperature_predictions': [],
            'temperature': [],
            'actions': []
        }
        self.logs = None

    def run(self) -> None:

        results_path = self._results_path + f'/control_{self.controller.__class__.__name__}/'
        self.create_result_folder(results_path)
        self.logs = get_log_filename(results_path)
        self.run_simulation()
        self.save_results(final=True)

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
            self.building.observe_weather(state)
            if self.iteration_counter > 0 and self.iteration_counter % (24 * self.energyplus_timesteps_in_hour) == 0:
                self.save_results(final=False)
            if self.iteration_counter > 0 and self.iteration_counter % self.controller.planning_timesteps == 0:
                current_date = self.building.send_current_datetime(self.iteration_counter)
                print(f'------------ date {current_date} ------------ \n')
                with open(self.logs, 'a') as f:
                    f.write(f"************************************************ \n")
                    f.write(f"********{current_date}******** \n")
                current_obs = self.building.observe_zones(n_lags=self.obs_lags)
                weather_forecast = self.building.send_weather_forecast(
                    state=state,
                    horizon=self.horizon,
                    iteration_counter=self.iteration_counter
                )
                weather_lags = self.building.send_weather_lags(n_lags=self.n_lags)
                action_lags = self.building.send_actions_lags(n_lags=self.n_lags + 1, heating=self.heating_period)
                time_to_start_dr_event, time_to_end_dr_event = self.send_time_to_dr_event(self.iteration_counter)
                power_schedule = self.read_consumption_schedule(self.iteration_counter)
                slacked_power_schedule = power_schedule * (1 - self.slack_power_constraint)
                temperature_targets = np.tile(
                    self.building.zones_temperature_setpoint[:, 1-int(self.heating_period)],  # Select heating / cooling setpoint
                    (self.horizon, 1)
                )
                setpoint_changes, power = self.controller.run(
                    observation=current_obs[self.zone_id],
                    weather=weather_forecast,
                    weather_lags=weather_lags,
                    action_lags=action_lags,
                    power_schedule=slacked_power_schedule,
                    temperature_targets=temperature_targets,
                    time_to_start_dr_event=time_to_start_dr_event,
                    time_to_end_dr_event=time_to_end_dr_event,
                    energyplus_timestep_duration=self.energyplus_timestep_duration,
                    logs_file=self.logs,
                )
                self.building.update_setpoint_changes(
                    setpoints_changes=np.expand_dims(setpoint_changes, 0),
                    controlled_zones_id=self.controlled_zones_id,
                    heating=self.heating_period
                )
                # Make predictions to compare with true values
                current_obs = self.building.observe_zones(n_lags=self.obs_lags)
                predictions = self.controller.make_predictions(
                    current_obs=current_obs[self.zone_id],
                    setpoint_changes=setpoint_changes,
                    weather_forecast=weather_forecast,
                    weather_lags=weather_lags,
                    action_lags=action_lags,
                    energyplus_timestep_duration=self.energyplus_timestep_duration,
                    prediction_horizon=self.controller.prediction_model.prediction_horizon
                )
                # update memory
                self.control_memory['power_schedule'].extend(
                    power_schedule[:self.controller.planning_timesteps].copy()
                )
                self.control_memory['slacked_power_schedule'].extend(
                    slacked_power_schedule[:self.controller.planning_timesteps].copy()
                )
                self.control_memory['power_predictions'].extend(
                    predictions['power'][:self.controller.planning_timesteps].copy()
                )
                self.control_memory['temperature_predictions'].extend(
                    predictions['temperature'][:self.controller.planning_timesteps].copy()
                )
                self.control_memory['actions'].extend(setpoint_changes[:self.controller.planning_timesteps].copy())

            # set the temperature set point with the change
            self.building.change_temperature_setpoints(state=state, zone_id=self.zone_id)

    def time_step_handler_end(self, state) -> None:
        """This method is executed at the end of each EnergyPlus iteration"""

        sys.stdout.flush()
        if not self.building.has_handles:
            self.building.get_handles(state)

        if self._done_warm_up:
            self.building.observe_temperature(state)
            self.building.observe_hvac_power(state)
            self.building.update_memory()
            self.iteration_counter += 1

    def save_results(self, final: bool) -> None:
        results_path = self._results_path + f'/control_{self.controller.__class__.__name__}/'
        self.building.save_memory(results_path, final=False)
        self.control_memory['power'] = np.vstack(
            self.building.memory[f'hvac_power_zone{self.zone_id}'][self.controller.planning_timesteps:]
        ).flatten().copy()
        self.control_memory['temperature'] = self.building.memory[f'temperature_zone{0}'][
        self.controller.planning_timesteps:].copy()
        control_memory = pd.DataFrame(self.control_memory)
        save_control_memory(control_memory, results_path, final=final)
