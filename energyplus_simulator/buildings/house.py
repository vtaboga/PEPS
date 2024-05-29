import numpy as np
import sys

from typing import List
from energyplus_simulator.buildings.energyplus_abstract_building import EnergyPlusAbstractBuilding


class House(EnergyPlusAbstractBuilding):
    """
    Wraps the house building
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
        self.zones_temperature_setpoint = np.array([[zones_temperature[0], zones_temperature[0]]])
        self._zones_fan_energy_sensor = []
        self._zones_heating_coil_energy_sensor = []
        self._zones_reheating_coil_energy_sensor = []
        self._zones_cooling_coil_energy_sensor = []
        self.zones_setpoint_changes = np.zeros((self.zones_number, 2))

    def get_handles(self, state: int) -> None:

        if self.api.exchange.api_data_fully_ready(state):
            # Sensors
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

            zone_full_name = self.get_zone_name_from_id(zid=0, complete=True)
            heating_temp_sp_sensor = self.api.exchange.get_variable_handle(
                state=state,
                variable_name="Zone Thermostat Heating Setpoint Temperature",
                variable_key=zone_full_name
            )
            if not self.heating_only:
                cooling_temp_sp_sensor = self.api.exchange.get_variable_handle(
                    state=state,
                    variable_name="Zone Thermostat Cooling Setpoint Temperature",
                    variable_key=zone_full_name
                )
            heating_coil_energy_sensor_name = self.get_heating_coil_meter_name_from_id(zid=0)
            heating_coil_energy_sensor = self.api.exchange.get_variable_handle(
                state=state,
                variable_name='Heating Coil Electricity Energy',
                variable_key=heating_coil_energy_sensor_name
            )
            if not self.heating_only:
                reheating_coil_energy_sensor_name = self.get_reheating_coil_meter_name_from_id(zid=0)
                reheating_coil_energy_sensor = self.api.exchange.get_variable_handle(
                    state=state,
                    variable_name='Heating Coil Electricity Energy',
                    variable_key=reheating_coil_energy_sensor_name
                )
                cooling_coil_energy_sensor_name = self.get_cooling_coil_meter_name_from_id(zid=0)
                cooling_coil_energy_sensor = self.api.exchange.get_variable_handle(
                    state=state,
                    variable_name='Cooling Coil Electricity Energy',
                    variable_key=cooling_coil_energy_sensor_name
                )
            fan_energy_sensor_name = self.get_fan_meter_name_from_id(zid=0)
            fan_energy_sensor = self.api.exchange.get_variable_handle(
                state=state,
                variable_name='Fan Electricity Energy',
                variable_key=fan_energy_sensor_name
            )
            # Actuators
            heating_temp_sp_key = 'Heating Setpoint'
            heating_temp_sp_actuator = self.api.exchange.get_actuator_handle(
                state=state,
                component_type='Schedule:Compact',
                control_type='Schedule Value',
                actuator_key=heating_temp_sp_key
            )
            if not self.heating_only:
                cooling_temp_sp_key = 'Cooling Setpoint'
                cooling_temp_sp_actuator = self.api.exchange.get_actuator_handle(
                    state=state,
                    component_type='Schedule:Compact',
                    control_type='Schedule Value',
                    actuator_key=cooling_temp_sp_key
                )

            self._zones_heating_setpoint_sensor.append(heating_temp_sp_sensor)
            self.zones_heating_setpoint_actuator.append(heating_temp_sp_actuator)
            self._zones_fan_energy_sensor.append(fan_energy_sensor)
            self._zones_heating_coil_energy_sensor.append(heating_coil_energy_sensor)

            if not self.heating_only:
                self._zones_cooling_setpoint_sensor.append(cooling_temp_sp_sensor)
                self.zones_cooling_setpoint_actuator.append(cooling_temp_sp_actuator)
                self._zones_cooling_coil_energy_sensor.append(cooling_coil_energy_sensor)
                self._zones_reheating_coil_energy_sensor.append(reheating_coil_energy_sensor)

            # get zone specific handlers
            for zid in range(self.zones_number):
                zone_full_name = self.get_zone_name_from_id(zid, complete=True)
                zone_temp_sensor = self.api.exchange.get_variable_handle(
                    state=state,
                    variable_name='Zone Mean Air Temperature',
                    variable_key=zone_full_name
                )
                self._zones_temperature_sensor.append(zone_temp_sensor)

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

        # Read coil electric energy consumption
        zones_coil_energy = np.array(
            [self.api.exchange.get_variable_value(state, sensor) for sensor in self._zones_heating_coil_energy_sensor]
        )
        if not self.heating_only:
            zones_reheating_coil_energy = np.array(
                [self.api.exchange.get_variable_value(state, sensor) for sensor in
                    self._zones_reheating_coil_energy_sensor]
            )
            zones_cooling_coil_energy = np.array(
                [self.api.exchange.get_variable_value(state, sensor) for sensor in
                    self._zones_cooling_coil_energy_sensor]
            )
            zones_coil_energy += zones_reheating_coil_energy + zones_cooling_coil_energy
        zones_coil_power = self._joule_to_kilowatt_per_system_timestep(zones_coil_energy, state)
        # Read fan electric energy consumption
        zones_fan_energy = np.array(
            [self.api.exchange.get_variable_value(state, fan_sensor) for fan_sensor in self._zones_fan_energy_sensor]
        )
        zones_fan_power = self._joule_to_kilowatt_per_system_timestep(zones_fan_energy, state)
        self.zones_hvac_power = zones_fan_power + zones_coil_power  # single value in a list because only one zone

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
            print('Problem with zones heating temperature setpoint sensor')
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

        if -1 in self._zones_heating_coil_energy_sensor:
            print('Problem with zones heating coil energy sensor')
            for i, j in enumerate(self._zones_heating_coil_energy_sensor):
                if j == -1:
                    print('problem with ', i, 'get ', j)
            sys.exit(1)

        if not self.heating_only:

            if -1 in self._zones_cooling_setpoint_sensor:
                print('Problem with zones heating temperature setpoint sensor')
                for i, j in enumerate(self._zones_cooling_setpoint_sensor):
                    if j == -1:
                        print('problem with ', i, 'get ', j)
                sys.exit(1)

            if -1 in self._zones_reheating_coil_energy_sensor:
                print('Problem with zones reheating coil energy sensor')
                for i, j in enumerate(self._zones_reheating_coil_energy_sensor):
                    if j == -1:
                        print('problem with ', i, 'get ', j)
                sys.exit(1)

            if -1 in self._zones_cooling_coil_energy_sensor:
                print('Problem with zones cooling coil energy sensor')
                for i, j in enumerate(self._zones_cooling_coil_energy_sensor):
                    if j == -1:
                        print('problem with ', i, 'get ', j)
                sys.exit(1)

        if -1 in self._zones_fan_energy_sensor:
            print('Problem with zones fan sensor')
            for i, j in enumerate(self._zones_fan_energy_sensor):
                if j == -1:
                    print('problem with ', i, 'get ', j)
            sys.exit(1)

        if -1 in self.zones_heating_setpoint_actuator:
            print('Problem with zones temp sp actuator')
            sys.exit(1)

        if not self.heating_only:
            if -1 in self.zones_cooling_setpoint_actuator:
                print('Problem with zones temp sp actuator')
                sys.exit(1)

    @staticmethod
    def get_zone_name_from_id(zid: int, complete: bool) -> str:
        zone_name_mapping = ['LIVING ZONE', 'GARAGE ZONE', 'ATTIC ZONE']
        return zone_name_mapping[zid]

    @staticmethod
    def get_fan_meter_name_from_id(zid: int) -> str:
        if zid == 0:
            fan_meter_mapping = 'Supply Fan'
        else:
            raise ValueError('Only zone 0 has a fan')
        return fan_meter_mapping

    @staticmethod
    def get_heating_coil_meter_name_from_id(zid: int) -> str:
        if zid == 0:
            coil_meter_mapping = 'Main Heating Coil'
        else:
            raise ValueError('Only zone 0 has a coil')
        return coil_meter_mapping

    @staticmethod
    def get_reheating_coil_meter_name_from_id(zid: int) -> str:
        if zid == 0:
            coil_meter_mapping = 'Supp Heating Coil'
        else:
            raise ValueError('Only zone 0 has a coil')
        return coil_meter_mapping

    @staticmethod
    def get_cooling_coil_meter_name_from_id(zid: int) -> str:
        if zid == 0:
            coil_meter_mapping = 'Cooling Coil'
        else:
            raise ValueError('Only zone 0 has a coil')
        return coil_meter_mapping

    @staticmethod
    def get_heating_energy_meter_name_from_id(zid: int) -> str:
        if zid == 0:
            hc_meter_mapping = 'Heating:EnergyTransfer:Zone:LIVING ZONE'
        else:
            raise ValueError('Only zone 0 has an heating energy meter')
        return hc_meter_mapping

    @staticmethod
    def get_zone_id_from_name(name: str) -> int:
        zone_id_mapping = {
            'LIVING ZONE': 0,
            'GARAGE': 1,
            'ATTIC ZONE': 2
        }
        return zone_id_mapping[name]
