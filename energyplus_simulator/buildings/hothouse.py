from energyplus_simulator.buildings.house import House
from typing import List


class HotHouse(House):
    """
    Wraps the house building
    Same as the basic house but in a hot climate zone.
    This wrapper helps separate results from the basic house
    """

    def __init__(
        self,
        heating_only: bool,
        zones_number: int,
        zones_temperature: List,
        energyplus_timesteps_in_hour: int
    ):

        super().__init__(
            energyplus_timesteps_in_hour=energyplus_timesteps_in_hour,
            zones_number=zones_number,
            zones_temperature=zones_temperature,
            heating_only=heating_only
        )
