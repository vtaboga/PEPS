import gin
import jax.numpy as jnp

from controllers.abstract_controller import ABCController
from typing import Dict


@gin.configurable
class RuleBasedController(ABCController):

    def __init__(
        self,
        prediction_model,
        preheating_period: int,
        planning_timesteps: int,
        lower_setpoint: float,
        upper_setpoint: float,
        mean_constants: Dict,
        std_constants: Dict,
        state_indexes: Dict,
        model_path: str
    ):
        ABCController.__init__(self)
        self.prediction_model = prediction_model(
            mean=mean_constants,
            std=std_constants,
            state_indexes=state_indexes,
            model_path=model_path
        )
        self.setpoint_change_index = state_indexes['setpoint_change']
        self.planning_timesteps = planning_timesteps
        self.prediction_horizon = self.prediction_model.prediction_horizon
        self.preheating_period = preheating_period
        self.lower_setpoint = lower_setpoint
        self.upper_setpoint = upper_setpoint
        self.action_taken = False

    def run(
        self,
        observation: jnp.array,
        weather: jnp.array,
        power_schedule: jnp.array,
        temperature_targets: jnp.array,
        time_to_start_dr_event,
        time_to_end_dr_event,
        energyplus_timestep_duration: int,
        logs_file: str,
    ):
        """
        :param observation:
        :param weather:
        :param power_schedule:
        :param temperature_targets:
        :param time_to_start_dr_event:
        :param time_to_end_dr_event:
        :param energyplus_timestep_duration:
        :param logs_file:
        :return:
        """

        # Predict the power consumption with no setpoint change
        power_schedule = self.process_power_schedule(
            observation=observation,
            weather=weather,
            power_schedule=power_schedule,
            energyplus_timestep_duration=energyplus_timestep_duration,
            logs_file=logs_file
        )

        if time_to_end_dr_event < 0:
            # Reset action_taken to false for the next day after the end of the dr event
            self.action_taken = False

        if self.action_taken or power_schedule is not None:
            setpoint_changes = jnp.concatenate(
                ([
                    jnp.ones(max(0, time_to_start_dr_event)) * self.upper_setpoint,  # preheating / pre cooling
                    jnp.ones(time_to_end_dr_event) * self.lower_setpoint
                ])
            )
            if setpoint_changes.shape[0] < self.prediction_horizon:
                setpoint_changes = jnp.concatenate(
                    [setpoint_changes, jnp.zeros(self.prediction_horizon - setpoint_changes.shape[0])]
                )
            else:
                setpoint_changes = setpoint_changes[:self.prediction_horizon]
            self.action_taken = True
        else:
            setpoint_changes = jnp.zeros(self.prediction_horizon)

        return setpoint_changes
