import numpy as np
import sys

from typing import List
from energyplus_simulator.buildings.energyplus_abstract_building import EnergyPlusAbstractBuilding


class LowriseApartments(EnergyPlusAbstractBuilding):
    """
    Wraps the midrise apartements building
    This class is used to specify the name of the variables, meters and actuators of the house_heating_only.idf file
    """

    def __init__(
        self,
        heating_only: bool,
        zones_number: int,
        zones_temperature: List,
        energyplus_timesteps_in_hour: int
    ):

        super().__init__(energyplus_timesteps_in_hour=energyplus_timesteps_in_hour, heating_only=heating_only)
        self.zones_number = zones_number
        self.zones_temperature = zones_temperature
        self.zones_temperature_setpoint = np.array([zones_temperature, zones_temperature]).transpose()
        self.zones_setpoint_changes = np.zeros((self.zones_number, 2))
        self._zones_coil_power_sensor = []

    def get_handles(self, state: int) -> None:

        if self.api.exchange.api_data_fully_ready(state):
            self._outdoor_temp_sensor = self.api.exchange.get_variable_handle(
                state=state,
                variable_name="Site Outdoor Air Drybulb Temperature",
                variable_key="Environment"
            )
            self._wind_speed_sensor = self.api.exchange.get_variable_handle(
                state=state,
                variable_name='Site Wind Speed',
                variable_key='Environment'
            )
            self._wind_direction_sensor = self.api.exchange.get_variable_handle(
                state=state,
                variable_name='Site Wind Direction',
                variable_key='Environment'
            )
            # get zone specific handlers
            for zid in range(self.zones_number):
                zone_full_name = self.get_zone_name_from_id(zid, complete=True)
                zone_temp_sensor = self.api.exchange.get_variable_handle(
                    state=state,
                    variable_name='Zone Mean Air Temperature',
                    variable_key=zone_full_name
                )
                zone_heating_temp_sp_sensor = self.api.exchange.get_variable_handle(
                    state=state,
                    variable_name="Zone Thermostat Heating Setpoint Temperature",
                    variable_key=zone_full_name
                )
                coil_electricity_meter_name = self.get_coil_meter_name_from_id(zid)
                zone_coil_power_sensor = self.api.exchange.get_meter_handle(
                    state=state,
                    meter_name=coil_electricity_meter_name
                )
                temp_sp_key = 'heating_sch_' + self.get_zone_name_from_id(zid, complete=False)
                zone_heating_temp_sp_actuator = self.api.exchange.get_actuator_handle(
                    state=state,
                    component_type='Schedule:Compact',
                    control_type='Schedule Value',
                    actuator_key=temp_sp_key
                )
                self._zones_temperature_sensor.append(zone_temp_sensor)
                self._zones_heating_setpoint_sensor.append(zone_heating_temp_sp_sensor)
                self._zones_coil_power_sensor.append(zone_coil_power_sensor)
                self.zones_heating_setpoint_actuator.append(zone_heating_temp_sp_actuator)
            self._check_handle_values()
            self.has_handles = True

    def observe_hvac_power(self, state) -> None:
        """
        :param state: state of the EnergyPlus simulation returned by the api state_manager
        :return: None

        Read the different meter and variable of EnergyPlus.
        Update the power and energy values of the building with the values of the current timestep.
        This method should be used at the end of a timestep
        !! The variable values depend on a simulation sub-timestep duration. It may lead to unexpected results.
           It is better to work with meter values when possible. !!
        """

        self.zone_timestep_duration = self.api.exchange.zone_time_step(state)
        self.system_timestep_duration = self.api.exchange.system_time_step(state)
        zones_heating_coil_energy = np.array(
            [self.api.exchange.get_meter_value(state, sensor) for sensor in self._zones_coil_power_sensor]
        )
        self.zones_hvac_power = self._joule_to_kilowatt_per_simulation_timestep(zones_heating_coil_energy)

    def _check_handle_values(self) -> None:

        if self._outdoor_temp_sensor == -1:
            print('Problem with a sensor handle')
            sys.exit(1)
        if self._wind_speed_sensor == -1:
            print('Problem with wind speed sensor')
            sys.exit(1)
        if self._wind_direction_sensor == -1:
            print('Problem with wind direction sensor')
            sys.exit(1)
        if -1 in self._zones_heating_setpoint_sensor:
            print('Problem with zones temperature setpoint sensor')
            for i, j in enumerate(self._zones_heating_setpoint_sensor):
                if j == -1:
                    print('problem with ', i, 'get ', j)
            sys.exit(1)
        if -1 in self._zones_temperature_sensor:
            print('Problem with zones temperature sensor')
            for i, j in enumerate(self._zones_temperature_sensor):
                if j == -1:
                    print('problem with ', i, 'get ', j)
            sys.exit(1)
        if -1 in self._zones_coil_power_sensor:
            print('Problem with zones heating coil meter sensor')
            for i, j in enumerate(self._zones_coil_power_sensor):
                if j == -1:
                    print('problem with ', i, 'get ', j)
            sys.exit(1)
        if -1 in self.zones_heating_setpoint_actuator:
            print('Problem with zones temp sp actuator')
            sys.exit(1)

    @staticmethod
    def get_zone_name_from_id(zid: int, complete: bool) -> str:

        if complete:
            zone_name_mapping = [
                'LIVING_UNIT1_FRONTROW_BOTTOMFLOOR',
                'LIVING_UNIT2_FRONTROW_BOTTOMFLOOR',
                'LIVING_UNIT3_FRONTROW_BOTTOMFLOOR',
                'LIVING_UNIT1_BACKROW_BOTTOMFLOOR',
                'LIVING_UNIT2_BACKROW_BOTTOMFLOOR',
                'LIVING_UNIT3_BACKROW_BOTTOMFLOOR',
                'LIVING_UNIT1_FRONTROW_MIDDLEFLOOR',
                'LIVING_UNIT2_FRONTROW_MIDDLEFLOOR',
                'LIVING_UNIT3_FRONTROW_MIDDLEFLOOR',
                'LIVING_UNIT1_BACKROW_MIDDLEFLOOR',
                'LIVING_UNIT2_BACKROW_MIDDLEFLOOR',
                'LIVING_UNIT3_BACKROW_MIDDLEFLOOR',
                'LIVING_UNIT1_FRONTROW_TOPFLOOR',
                'LIVING_UNIT2_FRONTROW_TOPFLOOR',
                'LIVING_UNIT3_FRONTROW_TOPFLOOR',
                'LIVING_UNIT1_BACKROW_TOPFLOOR',
                'LIVING_UNIT2_BACKROW_TOPFLOOR',
                'LIVING_UNIT3_BACKROW_TOPFLOOR', ]
        else:
            zone_name_mapping = [
                '1bf', '2bf', '3bf',
                '1bb', '2bb', '3bb',
                '1mf', '2mf', '3mf',
                '1mb', '2mb', '3mb',
                '1tf', '2tf', '3tf',
                '1tb', '2tb', '3tb']

        return zone_name_mapping[zid]

    @staticmethod
    def get_coil_meter_name_from_id(zid: int) -> str:

        coil_meter_mapping = [
            "Meter_Heating_Coil_Unit1_FrontRow_BottomFloor",
            "Meter_Heating_Coil_Unit2_FrontRow_BottomFloor",
            "Meter_Heating_Coil_Unit3_FrontRow_BottomFloor",
            "Meter_Heating_Coil_Unit1_BackRow_BottomFloor",
            "Meter_Heating_Coil_Unit2_BackRow_BottomFloor",
            "Meter_Heating_Coil_Unit3_BackRow_BottomFloor",
            "Meter_Heating_Coil_Unit1_FrontRow_MiddleFloor",
            "Meter_Heating_Coil_Unit2_FrontRow_MiddleFloor",
            "Meter_Heating_Coil_Unit3_FrontRow_MiddleFloor",
            "Meter_Heating_Coil_Unit1_BackRow_MiddleFloor",
            "Meter_Heating_Coil_Unit2_BackRow_MiddleFloor",
            "Meter_Heating_Coil_Unit3_BackRow_MiddleFloor",
            "Meter_Heating_Coil_Unit1_FrontRow_TopFloor",
            "Meter_Heating_Coil_Unit2_FrontRow_TopFloor",
            "Meter_Heating_Coil_Unit3_FrontRow_TopFloor",
            "Meter_Heating_Coil_Unit1_BackRow_TopFloor",
            "Meter_Heating_Coil_Unit2_BackRow_TopFloor",
            "Meter_Heating_Coil_Unit3_BackRow_TopFloor"
        ]

        return coil_meter_mapping[zid]

    @staticmethod
    def get_zone_id_from_name(name: str) -> int:

        zone_id_mapping = {
            '1bf': 0, '2bf': 1, '3bf': 2,
            '1bb': 3, '2bb': 4, '3bb': 5,
            '1mf': 6, '2mf': 7, '3mf': 8,
            '1mb': 9, '2mb': 10, '3mb': 11,
            '1tf': 12, '2tf': 13, '3tf': 14,
            '1tb': 15, '2tb': 16, '3tb': 17
        }

        return zone_id_mapping[name]
