import gin
import sys
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import ray
import json
import os
import time

from typing import List, Dict
from common.system import System
from common.configurations.gin_utils import query_gin_parameter
from controllers.shooting_controller import RayShootingController, ShootingController
from controllers.robust_controller import RobustController, RayRobustController
from common.utils import get_log_filename, save_control_memory
from off_line_computation.utils import load_norm_constants, get_processed_state_indexes


def load_ray_controller_parameters(
    zone_id,
    data_path,
    model_path,
    model_type,
    simulation_id,
    base_model_config,
    model_configuration,
    controller_type,
    controller_configuration
):

    mean, std = load_norm_constants(zone_id, data_path)
    state_indexes = get_processed_state_indexes(data_path)
    model_path += f'{model_type}/zone{zone_id}/train_test_{base_model_config}_model_{model_configuration}'
    controller_config_path = f'./controllers/configurations/{controller_type}/config{controller_configuration}.json'
    with open(controller_config_path, 'r') as f:
        controller_parameters = json.load(f)

    controller_parameters['mean_constants'] = mean
    controller_parameters['std_constants'] = std
    controller_parameters['state_indexes'] = state_indexes
    controller_parameters['model_path'] = model_path

    return controller_parameters


@gin.configurable
class MultiZoneBuildingSimulation(System):

    def __init__(
        self,
        simulation_id: int,
        control_experiment_id: int,
        building_type: str,
        number_of_zones: int,
        heating_only: bool,
        heating_period: bool,
        weather_file: str,
        zones_temperature: List[float],
        controlled_zones_id: List[int],
        observation_lags: int,
        prediction_horizon: int,
        controllers_type: List[str],
        controllers_configuration: List[int],
        base_model_config: int,
        models_type: List[str],
        models_class,
        models_configuration: List[str],
        coordinator_class,
        coordinator_config: int,
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
            zones_number=number_of_zones,
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

        self.heating_period = heating_period
        self._results_path = self._results_directory + f'/control_results/experiment_{control_experiment_id}'
        self.create_result_addm_folder(self._results_path)

        try:
            schedule_path = f'common/configurations/consumption_schedule/{consumption_schedule_name}.csv'
            self.consumption_schedule = pd.read_csv(schedule_path)
            self.consumption_schedule['date'] = pd.to_datetime(self.consumption_schedule['date'])
        except FileNotFoundError as e:
            print(f'could not load the consumption schedule, got error {e}')
            raise e

        # Instantiate zones' controllers
        self.controllers = {}
        data_path = f'common/results/{building_type}/simulation_{simulation_id}/processed_data'
        model_path = f'common/results/{building_type}/simulation_{simulation_id}/models/'
        self.controlled_zones_id = controlled_zones_id
        for i, zone_id in enumerate(controlled_zones_id):
            mean, std = load_norm_constants(zone_id, data_path)
            state_indexes = get_processed_state_indexes(data_path)
            parameters = load_ray_controller_parameters(
                zone_id=zone_id,
                data_path=data_path,
                model_path=model_path,
                model_type=models_type[i],
                simulation_id=simulation_id,
                model_configuration=models_configuration[i],
                base_model_config=base_model_config,
                controller_type=controllers_type[i],
                controller_configuration=controllers_configuration[i]
            )
            prediction_model = models_class[i]
            parameters['prediction_model'] = prediction_model(
                zone_id=zone_id,
                mean=mean,
                std=std,
                state_indexes=state_indexes,
                model_path=parameters['model_path']
            )
            if controllers_type[i] == 'RayShootingController':
                self.controllers[f'{zone_id}'] = RayShootingController.remote(**parameters)
            elif controllers_type[i] == 'RayRobustController':
                self.controllers[f'{zone_id}'] = RayRobustController.remote(**parameters)
            else:
                print(f'controller type {controllers_type[i]} for controller {i} not handled.')
                raise ValueError

        self.n_lags = parameters['prediction_model'].n_lags
        # Instantiate coordinator
        self.coordinator = coordinator_class(number_of_zones=len(controlled_zones_id))
        # Logs
        self.control_memory = None
        self.reset_control_memory()

    def run(self) -> None:

        self.logs = get_log_filename(self._results_path)
        self.run_simulation()
        self.save_control_memory(final=True)

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

        sys.stdout.flush()
        if not self.building.has_handles:
            self.building.get_handles(state)

        if self._done_warm_up:
            self.building.observe_weather(state)
            if self.iteration_counter >= self.energyplus_timesteps_in_hour and self.iteration_counter % (
                    24 * self.energyplus_timesteps_in_hour) == 0:
                # save partial results every day
                self.save_control_memory(final=False)

            if self.iteration_counter >= 2 * self.energyplus_timesteps_in_hour and self.iteration_counter % self.coordinator.planning_timesteps == 0:
                current_date = self.building.send_current_datetime(self.iteration_counter)
                print(f'------------ date {current_date} ------------ \n')
                with open(self.logs, 'a') as f:
                    f.write(f"************************************************ \n")
                    f.write(f"********{current_date}******** \n")
                current_observations = self.building.observe_zones(n_lags=self.obs_lags)
                weather_lags = self.building.send_weather_lags(n_lags=self.n_lags)
                # take one extra lag to compute the setpoint difference
                action_lags = self.building.send_actions_lags(n_lags=self.n_lags + 1, heating=self.heating_period)
                weather_forecast = self.building.send_weather_forecast(
                    state=state,
                    horizon=self.horizon,
                    iteration_counter=self.iteration_counter
                )
                power_schedule = self.read_consumption_schedule(self.iteration_counter)
                slacked_power_schedule = power_schedule * (1 - self.coordinator.slack_maximum_power)

                # Compute new setpoint changes for each controlled zone
                setpoints_changes = self.distributed_optimization(
                    current_observations=current_observations,
                    weather_forecast=weather_forecast,
                    weather_lags=weather_lags,
                    action_lags=action_lags,
                    power_schedule=slacked_power_schedule
                )

                # Apply setpoints changes
                self.building.update_setpoint_changes(
                    setpoints_changes=setpoints_changes,
                    controlled_zones_id=self.controlled_zones_id,
                    heating=self.heating_period
                )
            for zone_id in self.controlled_zones_id:
                self.building.change_temperature_setpoints(state=state, zone_id=zone_id)

    def time_step_handler_end(self, state) -> None:

        sys.stdout.flush()
        if not self.building.has_handles:
            self.building.get_handles(state)

        if self._done_warm_up:
            self.building.observe_temperature(state)
            self.building.observe_hvac_power(state)
            self.building.update_memory()
            self.iteration_counter += 1

    def distributed_optimization(
        self,
        current_observations: jnp.array,
        weather_forecast: jnp.array,
        weather_lags: jnp.array,
        action_lags: jnp.array,
        power_schedule: jnp.array
    ) -> jnp.array:
        """
        :param current_observations: shape (n_zones, n_lags, obs_dim)
        :param weather_forecast: shape (horizon, weather_dim)
        :param weather_lags:
        :param action_lags:
        :param power_schedule:
        :return: setpoints changes for each zone on the next horizon

        Execute ADMM
        """

        start_time = time.time()

        no_setpoint_changes = np.zeros((len(self.controlled_zones_id), self.horizon))
        min_setpoint_changes = np.vstack(
            [np.ones(self.horizon) * ray.get(self.controllers[f'{i}'].get_minimum_setpoint_change.remote()) for i in
                self.controlled_zones_id]
        )

        minimum_power_consumption = self.get_total_power_consumption_prediction(
            current_observations=current_observations,
            weather_forecast=weather_forecast,
            action_lags=action_lags,
            setpoints_changes=min_setpoint_changes,
            weather_lags=weather_lags
        )
        no_change_power_consumption = self.get_total_power_consumption_prediction(
            current_observations=current_observations,
            weather_lags=weather_lags,
            action_lags=action_lags,
            setpoints_changes=no_setpoint_changes,
            weather_forecast=weather_forecast
        )

        non_zero_mask = power_schedule != 0

        if (no_change_power_consumption[non_zero_mask] <= power_schedule[non_zero_mask]).all():
            setpoints_changes = no_setpoint_changes
            total_power_prediction = no_change_power_consumption
            n_admm_iterations = 0
        elif (minimum_power_consumption[non_zero_mask] >= power_schedule[non_zero_mask]).all():
            setpoints_changes = min_setpoint_changes
            total_power_prediction = minimum_power_consumption
            n_admm_iterations = 0
        else:
            # Define power targets to be the power consumption with no change if it stays below the scheduled maximum
            power_targets = np.where(
                non_zero_mask,
                np.minimum(power_schedule, no_change_power_consumption),
                no_change_power_consumption
            )
            admm_results = self.admm(
                power_targets=power_targets,
                zones_observation=current_observations,
                weather_forecast=weather_forecast,
                weather_lags=weather_lags,
                action_lags=action_lags
            )
            setpoints_changes = admm_results['zones_setpoint_changes']
            total_power_prediction = admm_results['total_power_prediction']
            n_admm_iterations = admm_results['n_iterations']

        stop_time = time.time()

        self.update_control_memory(
            total_power_prediction=total_power_prediction,
            zones_setpoint_changes=setpoints_changes,
            power_schedule=power_schedule,
            computation_time=stop_time - start_time,
            n_admm_iterations=n_admm_iterations
        )
        return setpoints_changes

    def admm(
        self,
        power_targets: jnp.array,
        zones_observation: np.array,
        weather_forecast: np.array,
        weather_lags,
        action_lags
    ) -> Dict:
        """
        :param power_targets: total building power consumption targets for the next horizon (shape (horizon,))
        :param zones_observation: current observations from each zone (shape (zones_number, horizon))
        :param weather_forecast: weather forecast for the next horizon
        :return:
        """

        def iteration():
            dual_variables = self.coordinator.send_dual_variables()
            targets = self.coordinator.send_power_targets()
            controllers_plan = self.get_new_setpoint_changes(
                current_observations=zones_observation,
                weather_forecast=weather_forecast,
                power_targets=targets,
                dual_variables=dual_variables,
                weather_lags=weather_lags,
                action_lags=action_lags
            )
            # print('controllers plan')
            # print(controllers_plan)
            has_converge = self.coordinator.run(controllers_plan['power'])
            return controllers_plan, has_converge

        # Initialization
        self.coordinator.reset()
        for zid in self.controlled_zones_id:
            ray.get(self.controllers[f'{zid}'].reset_controller.remote())
        self.coordinator.update_total_power(power_targets)
        controllers_plan, has_converge = iteration()
        n_iters = 1

        while not has_converge:
            controllers_plan, has_converge = iteration()
            n_iters += 1
            if n_iters > self.coordinator.maximum_admm_iterations:
                break

        total_power_forecast = jnp.sum(controllers_plan['power'], axis=0)
        self.coordinator.save_memory(
            path=self._results_path,
            iteration=self.iteration_counter
        )
        results = {
            'zones_setpoint_changes': controllers_plan['setpoints'],
            'total_power_prediction': total_power_forecast,
            'n_iterations': n_iters
        }

        return results

    def get_total_power_consumption_prediction(
        self,
        current_observations: jnp.array,
        setpoints_changes: jnp.array,
        weather_forecast: jnp.array,
        weather_lags: jnp.array,
        action_lags: jnp.array
    ) -> jnp.array:
        """
        :param current_observations:
        :param setpoints_changes:
        :param weather_forecast:
        :return:

        Make power consumption predictions in each controlled zone and sum to total
        """

        task_ids = [self.controllers[f'{zone_id}'].make_predictions.remote(
            current_obs=current_observations[zone_id],
            setpoint_changes=setpoints_changes[zone_id],
            weather_forecast=weather_forecast,
            weather_lags=weather_lags,
            action_lags=action_lags[zone_id],
            energyplus_timestep_duration=self.energyplus_timestep_duration,
            prediction_horizon=self.horizon
        ) for zone_id in self.controlled_zones_id]
        results = ray.get(task_ids)
        zones_power_predictions = np.array([zone_predictions['power'] for zone_predictions in results])
        total_power_predictions = np.sum(zones_power_predictions, axis=0)

        return total_power_predictions

    def get_new_setpoint_changes(
        self,
        current_observations: jnp.array,
        weather_forecast: jnp.array,
        weather_lags: jnp.array,
        action_lags: jnp.array,
        power_targets: jnp.array,
        dual_variables: jnp.array
    ):
        time_to_start_dr_event, time_to_end_dr_event = self.send_time_to_dr_event(self.iteration_counter)
        task_ids = [self.controllers[f'{zone_id}'].run.remote(
            current_observations[zone_id],
            weather_forecast,
            weather_lags,
            action_lags[zone_id],
            power_targets[:, zone_id],
            None,
            time_to_start_dr_event,
            time_to_end_dr_event,
            self.energyplus_timestep_duration,
            self.logs,
            dual_variables[:, zone_id],
            self.coordinator.rho
        ) for zone_id in self.controlled_zones_id]
        results = ray.get(task_ids)
        new_setpoint_changes, power_forecasts = zip(*results)
        controllers_plan = {'setpoints': jnp.array(new_setpoint_changes), 'power': jnp.array(power_forecasts)}

        return controllers_plan  # shape n_zones * horizon

    def reset_control_memory(self) -> None:

        self.control_memory = {
            'slacked_power_schedule': [],
            'total_power_prediction': [],
            'computation_time': [],
            'admm_iterations': []
        }
        for zone_id in self.controlled_zones_id:
            self.control_memory[f'setpoint_change_zone{zone_id}'] = []

    def update_control_memory(
        self,
        total_power_prediction,
        power_schedule,
        zones_setpoint_changes,
        n_admm_iterations,
        computation_time
    ) -> None:

        planning_steps = self.coordinator.planning_timesteps
        self.control_memory['total_power_prediction'].extend(total_power_prediction[:planning_steps].tolist())
        self.control_memory['slacked_power_schedule'].extend(power_schedule[:planning_steps].tolist())
        # ADMM is not necessarily done at every simulation step
        n_iterations = [n_admm_iterations] + [0] * (planning_steps - 1)
        computation_times = [computation_time] + [0] * (planning_steps - 1)
        self.control_memory['admm_iterations'].extend(n_iterations)
        self.control_memory['computation_time'].extend(computation_times)
        for zid in self.controlled_zones_id:
            self.control_memory[f'setpoint_change_zone{zid}'].extend(
                zones_setpoint_changes[zid, :planning_steps].tolist()
            )

    def save_control_memory(self, final: bool) -> None:

        self.building.save_memory(self._results_path, final=final)
        for key, value in self.control_memory.items():
            print(f"Length of '{key}': {len(value)}")
        control_memory = pd.DataFrame(self.control_memory)
        if final:
            if os.path.exists(self._results_path + "/temp_control_results.csv"):
                os.remove(self._results_path + "/temp_control_results.csv")
            control_memory.to_csv(self._results_path + "/control_results.csv")
        else:
            control_memory.to_csv(self._results_path + "/temp_control_results.csv")

    @staticmethod
    def create_result_addm_folder(results_path: str) -> None:
        if not os.path.exists(results_path + '/data'):
            os.makedirs(results_path + '/data')
        if not os.path.exists(results_path + '/admm_logs'):
            os.makedirs(results_path + '/admm_logs')
