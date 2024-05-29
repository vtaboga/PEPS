import gin
import jax.numpy as jnp
import jax
import time
import numpy as np
import ray

from controllers.abstract_controller import ABCController
from controllers.utils.optimizers import projected_gradient_descent
from typing import Dict, List, Union


@gin.configurable
class ShootingController(ABCController):

    def __init__(
        self,
        zone_id: int,
        prediction_model,
        model_path: str,
        planning_timesteps: int,
        temperature_weight: float,
        projected_gradient_params: Dict,
        mininum_setpoint_change: float,
        maximum_setpoint_change: float,
        setpoint_change_step: float,
        mean_constants: Dict,
        std_constants: Dict,
        state_indexes: Dict
    ):
        """
        :param prediction_model: prediction model class
        :param planning_timesteps: number of simulation timestep between two replanning.
        :param temperature_weight: weight of the temperature objective on the loss function
        :param projected_gradient_params: parameters of the projected gradient descent.
               see controllers.utils.optimizers
        :param setpoint_change_index: position on the action in the actions array
        :param action_normalization_constants: Dictionary containing the
               setpoint change mean and std on the training data
        """
        ABCController.__init__(self)

        # Instantiate the prediction model class
        if isinstance(prediction_model, type):
            self.prediction_model = prediction_model(
                zone_id=zone_id,
                mean=mean_constants,
                std=std_constants,
                state_indexes=state_indexes,
                model_path=model_path
            )
        else:
            self.prediction_model = prediction_model
        self.setpoint_change_index = state_indexes['setpoint_change']
        self.planning_timesteps = planning_timesteps
        self.mininum_setpoint_change = mininum_setpoint_change
        self.maximum_setpoint_change = maximum_setpoint_change
        self.setpoint_change_step = setpoint_change_step
        self.prediction_horizon = self.prediction_model.prediction_horizon
        self.projected_gradient_params = projected_gradient_params
        self.temperature_weight = temperature_weight

    @staticmethod
    def log_barrier(predictions, targets):
        difference = jnp.maximum(targets - predictions, 1e-40)
        error = jax.nn.softplus(- jnp.log(difference))
        return jnp.sum(error)

    def run(
        self,
        observation: jnp.array,
        weather: jnp.array,
        weather_lags,
        action_lags,
        power_schedule: jnp.array,
        temperature_targets: Union[None, jnp.array],
        time_to_start_dr_event,
        time_to_end_dr_event,
        energyplus_timestep_duration: int,
        logs_file: str,
        dual_variables: jnp.array = None,
        rho: float = None
    ) -> jnp.array:

        """
        :param observation: inputs of the model. May contain the lags of observation
        :param weather: weather and calendar info for the upcoming horizon
        :param power_schedule: power targets to track. Not normalized
        :param temperature_targets: temperature set points set by the user
        :param energyplus_timestep_duration: EnergyPlus simulation time step
        :param logs_file: path of the txt logs file to store projected gradient logs
        :param dual_variables: dual variables of the zone sent by the coordinator
        :param rho: Augmented Lagrangian parameter
        :return: An array containing the actions plan for the horizon on the zone (not normalized)
        """

        power_schedule = self.process_power_schedule(
            observation=observation,
            weather=weather,
            weather_lags=weather_lags,
            action_lags=action_lags,
            power_schedule=power_schedule,
            energyplus_timestep_duration=energyplus_timestep_duration,
            logs_file=logs_file
        )

        if power_schedule is None:
            # No changes if there is no maximum constraint on the horizon
            best_setpoint_changes = jnp.zeros(self.prediction_horizon)
            predictions = self.make_predictions(
                current_obs=observation,
                setpoint_changes=best_setpoint_changes,
                weather_forecast=weather,
                weather_lags=weather_lags,
                action_lags=action_lags,
                energyplus_timestep_duration=energyplus_timestep_duration,
                prediction_horizon=self.prediction_horizon
            )
        else:
            # check if there is a feasible solution
            actions_to_check = [jnp.float64(-i) for i in range(1, jnp.abs(self.mininum_setpoint_change) + 1)]
            check_passed = False
            for action in actions_to_check:
                init_sp_changes = jnp.full((self.prediction_horizon // self.planning_timesteps,), action)
                setpoints = jnp.repeat(init_sp_changes, self.planning_timesteps)
                predictions = self.make_predictions(
                    current_obs=observation,
                    setpoint_changes=setpoints,
                    weather_forecast=weather,
                    weather_lags=weather_lags,
                    action_lags=action_lags,
                    energyplus_timestep_duration=energyplus_timestep_duration,
                    prediction_horizon=self.prediction_horizon
                )
                if jnp.all(predictions['power'] <= power_schedule):
                    check_passed = True
                    with open(logs_file, 'a') as f:
                        f.write(
                            f"Pre check passed: for action {action} \n"
                            f"Predicted states {predictions} \n"
                            f"Targets {power_schedule} \n"
                        )
                    break

            if not check_passed:
                setpoint_changes = jnp.repeat(init_sp_changes, self.planning_timesteps)
                predictions = self.make_predictions(
                    current_obs=observation,
                    setpoint_changes=setpoint_changes,
                    weather_forecast=weather,
                    weather_lags=weather_lags,
                    action_lags=action_lags,
                    energyplus_timestep_duration=energyplus_timestep_duration,
                    prediction_horizon=self.prediction_horizon
                )

                with open(logs_file, 'a') as f:
                    f.write(
                        f"Pre check failed: no feasible solution for action {action} \n"
                        f"Predicted temperature {predictions} \n"
                        f"Targets {power_schedule} \n"
                    )
                return setpoint_changes, predictions['power']

            power_targets = jnp.where(jnp.isinf(power_schedule), 100.0, power_schedule)

            def error_func(setpoint_changes):
                setpoint_changes = jnp.repeat(setpoint_changes, self.planning_timesteps)
                predictions = self.make_predictions(
                    current_obs=observation,
                    setpoint_changes=setpoint_changes,
                    weather_forecast=weather,
                    weather_lags=weather_lags,
                    action_lags=action_lags,
                    energyplus_timestep_duration=energyplus_timestep_duration,
                    prediction_horizon=self.prediction_horizon
                )
                error_power = self.log_barrier(predictions['power'], power_targets)
                error_temperature = jnp.sum(jnp.square((temperature_targets - predictions['temperature'])))
                error = error_power + error_temperature
                return error

            start_time = time.time()
            # initialize at min action to make sure the first solution is feasible
            setpoint_changes, _ = projected_gradient_descent(
                loss_func=error_func,
                x0=init_sp_changes,
                lower_bound=self.mininum_setpoint_change,
                upper_bound=self.maximum_setpoint_change,
                iterations_number=self.projected_gradient_params['max_iterations_number'],
                line_search_alpha=self.projected_gradient_params['alpha'],
                line_search_beta=self.projected_gradient_params['beta'],
                line_search_maximum_iterations=self.projected_gradient_params['line_search_max_iterations'],
                logs_file=logs_file
            )
            end_time = time.time()
            total_execution_time = end_time - start_time

            # logs the errors for the best action
            setpoint_changes = jnp.repeat(setpoint_changes, self.planning_timesteps)
            predictions = self.make_predictions(
                current_obs=observation,
                setpoint_changes=setpoint_changes,
                weather_forecast=weather,
                weather_lags=weather_lags,
                action_lags=action_lags,
                energyplus_timestep_duration=energyplus_timestep_duration,
                prediction_horizon=self.prediction_horizon
            )
            best_error_power = self.log_barrier(predictions['power'], power_targets)
            if temperature_targets is not None:
                best_error_temperature = jnp.sum(jnp.square(self.temperature_weight * (temperature_targets - predictions['temperature'])))
            else:
                best_error_temperature = None

            with open(logs_file, 'a') as f:
                f.write(f"Total Execution Time for shooting_control: {total_execution_time:.2f} seconds\n")
                f.write(f"Error Power for Best Action: {best_error_power}\n")
                f.write(f"Error Temperature for Best Action: {best_error_temperature}\n")
                f.write(f'Maximum powers {power_targets.flatten()} \n')
                f.write(f'Predictions Power {predictions["power"]} \n')
                f.write(f'Prediction Temperature {predictions["temperature"]} \n')

            # round actions
            best_setpoint_changes = np.round(setpoint_changes / self.setpoint_change_step) * self.setpoint_change_step

        return best_setpoint_changes, predictions['power']


@ray.remote
class RayShootingController(ShootingController):

    def __init__(
        self,
        zone_id,
        prediction_model,
        model_path: str,
        planning_timesteps: int,
        temperature_weight: float,
        projected_gradient_params: Dict,
        mininum_setpoint_change: float,
        maximum_setpoint_change: float,
        setpoint_change_step: float,
        mean_constants: Dict,
        std_constants: Dict,
        state_indexes: Dict
    ):

        jax.config.update("jax_enable_x64", True)

        ShootingController.__init__(
            self,
            zone_id=zone_id,
            prediction_model=prediction_model,
            model_path=model_path,
            planning_timesteps=planning_timesteps,
            temperature_weight=temperature_weight,
            projected_gradient_params=projected_gradient_params,
            mininum_setpoint_change=mininum_setpoint_change,
            maximum_setpoint_change=maximum_setpoint_change,
            setpoint_change_step=setpoint_change_step,
            mean_constants=mean_constants,
            std_constants=std_constants,
            state_indexes=state_indexes
        )
        self.norm_temperature_obj = jnp.linalg.norm(jnp.ones(self.prediction_horizon) * self.mininum_setpoint_change)**2


    def run(
        self,
        observation: jnp.array,
        weather: jnp.array,
        weather_lags,
        action_lags,
        power_schedule: jnp.array,
        temperature_targets: Union[None, jnp.array],
        time_to_start_dr_event,
        time_to_end_dr_event,
        energyplus_timestep_duration: int,
        logs_file: str,
        dual_variables: jnp.array = None,
        rho: float = None
    ) -> jnp.array:

        """
        :param observation: inputs of the model. May contain the lags of observation
        :param weather: weather and calendar info for the upcoming horizon
        :param power_targets: power targets to track. Not normalized
        :param temperature_targets: temperature set points set by the user
        :param energyplus_timestep_duration: EnergyPlus simulation time step
        :param logs_file: path of the txt logs file to store projected gradient logs
        :param dual_variables: dual variables of the zone sent by the coordinator
        :param rho: Augmented Lagrangian parameter
        :return: An array containing the actions plan for the horizon on the zone (not normalized)
        """

        def error_func(setpoint_changes):
            setpoint_changes = jnp.repeat(setpoint_changes, self.planning_timesteps)
            predictions = self.make_predictions(
                current_obs=observation,
                setpoint_changes=setpoint_changes,
                weather_forecast=weather,
                weather_lags=weather_lags,
                action_lags=action_lags,
                energyplus_timestep_duration=energyplus_timestep_duration,
                prediction_horizon=self.prediction_horizon
            )
            error_temperature = jnp.linalg.norm(setpoint_changes)**2
            error_power = jnp.dot(dual_variables, power_schedule - predictions['power'])
            error_power += rho * jnp.linalg.norm(power_schedule - predictions['power'])**2
            error = 0.1 * error_temperature / self.norm_temperature_obj + error_power
            return error

        start_time = time.time()
        init_sp_changes = -jnp.ones((self.prediction_horizon // self.planning_timesteps))
        setpoint_changes, _ = projected_gradient_descent(
            loss_func=error_func,
            x0=init_sp_changes,
            lower_bound=self.mininum_setpoint_change,
            upper_bound=self.maximum_setpoint_change,
            iterations_number=self.projected_gradient_params['max_iterations_number'],
            line_search_alpha=self.projected_gradient_params['alpha'],
            line_search_beta=self.projected_gradient_params['beta'],
            line_search_maximum_iterations=self.projected_gradient_params['line_search_max_iterations'],
            logs_file=logs_file
        )
        end_time = time.time()
        total_execution_time = end_time - start_time

        # logs the errors for the best action
        setpoint_changes = jnp.repeat(setpoint_changes, self.planning_timesteps)
        predictions = self.make_predictions(
            current_obs=observation,
            setpoint_changes=setpoint_changes,
            weather_forecast=weather,
            weather_lags=weather_lags,
            action_lags=action_lags,
            energyplus_timestep_duration=energyplus_timestep_duration,
            prediction_horizon=self.prediction_horizon
        )

        error_temperature = jnp.linalg.norm(setpoint_changes) ** 2 / self.norm_temperature_obj
        error_power = jnp.dot(dual_variables, power_schedule - predictions['power'])
        error_power += rho * jnp.linalg.norm(power_schedule - predictions['power']) ** 2

        with open(logs_file, 'a') as f:
            f.write(f"Total Execution Time for shooting_control: {total_execution_time:.2f} seconds\n")
            f.write(f'Power targts {power_schedule.flatten()} \n')
            f.write(f'Predictions Power {predictions["power"]} \n')
            f.write(f'Prediction Temperature {predictions["temperature"]} \n')
            f.write(f'error temperature {np.round(error_temperature, 6)} \n')
            f.write(f'error power {np.round(error_power, 6)} \n')

        # round actions
        best_setpoint_changes = np.round(setpoint_changes / self.setpoint_change_step) * self.setpoint_change_step

        return best_setpoint_changes, predictions['power']

    def get_minimum_setpoint_change(self) -> float:
        return self.mininum_setpoint_change

    def get_maximum_setpoint_change(self) -> float:
        return self.maximum_setpoint_change

