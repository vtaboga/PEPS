import numpy as np

from energyplus_simulator.buildings.lowrise_apartements import LowriseApartments
from energyplus_simulator.buildings.house import House
from energyplus_simulator.buildings.hothouse import HotHouse
from typing import List


class EPlusWrapper:
    """
    Wraps a building from energyplus_simulator.buildings
    """

    def __init__(
        self,
        results_path: str,
        weather_path: str,
        building_path: str,
        building_type: str,
        heating_only: bool,
        zones_number: int,
        zones_temperature: List[float],
        energyplus_timesteps_in_hour: int,
        warmup_iterations: int
    ):
        """
        :param results_path: relative path (from the root where main.py is) to the results directory
        :param weather_path: relative path (from the root where main.py is) to the weather file (.epw format)
        :param building_path: relative path (from the root where main.py is) to the building desciption (.idf format)
        :param building_type: name of the building to run the simulation on (must be the name of one of the idf files)
        :param heating_only: Is the building equipped with a cooling system
        :param zones_number: number of zones in the building
        :param zones_temperature: initial temperature set-point of each zone in the building
        :param energyplus_timesteps_in_hour: number of timesteps per hour in EnergyPlus
        :param warmup_iterations: number of simulation warmup iteration (might change from one building to the other)
        """

        building_parameters = {
            'heating_only': heating_only,
            'zones_number': zones_number,
            'zones_temperature': zones_temperature,
            'energyplus_timesteps_in_hour': energyplus_timesteps_in_hour
        }
        self.building = self._create_building(building_type, building_parameters)
        self.current_time = 0

        self.zones_number = zones_number
        self._change_setpoints_counter = np.zeros(self.building.zones_number)
        self._setpoint_changes_reset = np.ones(self.building.zones_number)

        self._results_path = results_path
        self._weather_path = weather_path
        self._bldg_path = building_path

        self._done_warm_up = False
        self._total_warmup_iterations = warmup_iterations
        self._warm_up_count = 0

    @staticmethod
    def _create_building(building_type: str, building_parameters: dict):
        """
        :param building_type: type of building to instantiate
        :param building_parameters: parameters required to instantiate the building class
        :return: One of the building classes from energyplus_simulator.buildings
        """
        if building_type == 'house':
            return House(**building_parameters)
        elif building_type == 'hothouse':
            return HotHouse(**building_parameters)
        elif building_type == 'lowrise_apartments':
            return LowriseApartments(**building_parameters)
        else:
            raise ValueError(f"Invalid building type: {building_type}")
