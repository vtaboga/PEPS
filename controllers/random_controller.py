import gin
import numpy as np
import datetime

from typing import Dict


@gin.configurable
class RandomController:
    """
    Random temperature setpoint controller for a single zone
    """

    def __init__(
        self,
        num_timesteps_in_hour: int,
        maximum_duration: int,
        minimum_duration: int,
        mininum_setpoint_change: int,
        maximum_setpoint_change: int,
        heating_schedule: Dict,
        setpoint_change_step: float,
        off_season_change: float
    ):
        """
        :param num_timesteps_in_hour: EnergyPlus number of timesteps in one hour
        :param maximum_duration: maximum number of hours to keep the same temperature setpoint
        :param minimum_duration: minimum number of hours to keep the same temperature setpoint
        :param mininum_setpoint_change: minimum set-point change from the zone's default setpoint (in °C)
        :param maximum_setpoint_change: maximum set-point change from the zone's default setpoint (in °C)
        :param setpoint_change_step: increment of the thermostat (in °C)
        :param heating_schedule: dictionary with entry end_heating and end_cooling if format "day/month"
        :param off_season_change: set point change for the cooling season if currently in heating season and vice versa
                                  should be set to a large value to prevent heating in cooling season
        """

        self.num_timesteps_in_hour = num_timesteps_in_hour
        self.default_maximum_duration = maximum_duration
        self.default_minimum_duration = minimum_duration
        self._change_setpoints_counter = 0
        self._setpoint_change_reset = 0
        self._action_space = np.linspace(
            mininum_setpoint_change,
            maximum_setpoint_change,
            num=int((maximum_setpoint_change - mininum_setpoint_change) / setpoint_change_step) + 1
        )
        self.draw_setpoint_duration(
            minimum_duration=self.default_minimum_duration,
            maximum_duration=self.default_maximum_duration
        )
        self.initial_setpoint_change = np.zeros(2)  # heating and cooling set points
        self.current_setpoint_change = np.zeros(2)
        self.off_season_change = off_season_change
        self._heating_index = 0
        self._cooling_index = 1
        heating_schedule["start_heating"] = datetime.datetime.strptime(heating_schedule["start_heating"], "%d/%m")
        heating_schedule["end_heating"] = datetime.datetime.strptime(heating_schedule["end_heating"], "%d/%m")
        self.heating_schedule = heating_schedule

    def draw_setpoint_duration(self, minimum_duration: int, maximum_duration: int) -> None:
        """
        :param minimum_duration: minimum number of hours to keep the same temperature setpoint
        :param maximum_duration: maximum number of hours to keep the same temperature setpoint
        :return: None

        Call this method to draw a duration for a temperature setpoint.
        """
        self._setpoint_change_reset = np.random.randint(
            low=minimum_duration * self.num_timesteps_in_hour,
            high=maximum_duration * self.num_timesteps_in_hour
        )

    def random_setpoint_changes(
        self,
        date: datetime.datetime,
        minimum_duration: int = None,
        maximum_duration: int = None
    ) -> None:
        """
        :param minimum_duration: minimum number of hours to keep the same temperature set-point
        :param maximum_duration: maximum number of hours to keep the same temperature set-point
        :param date: current simulation date
        :return: None

        Call this method to draw a new temperature setpoint change for the zone being controlled.
        Calling this method only draw an action. The action need to be applied in the system (c.f. system files)
        """

        maximum_duration = maximum_duration if maximum_duration is not None else self.default_maximum_duration
        minimum_duration = minimum_duration if minimum_duration is not None else self.default_minimum_duration
        self._change_setpoints_counter += 1
        if self._change_setpoints_counter == self._setpoint_change_reset:
            self._change_setpoints_counter = 0
            self.draw_setpoint_duration(minimum_duration=minimum_duration, maximum_duration=maximum_duration)
            setpoint_change = np.random.choice(self._action_space)
            if self.is_heating_period(date):
                self.current_setpoint_change = np.array([setpoint_change, self.off_season_change])
            else:
                self.current_setpoint_change = np.array([-self.off_season_change, setpoint_change])

    def get_current_setpoint_change(self) -> np.array:
        """
        :return: The current setpoint change (in °C)
        """
        return self.current_setpoint_change

    def is_heating_period(self, current_date: datetime.datetime) -> bool:
        """
        :param current_date: date of the simulation
        :return: a boolean to specify if the date is in a heating period
        """

        return not (self.heating_schedule["end_heating"] < current_date < self.heating_schedule["start_heating"])
