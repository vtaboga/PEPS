import gin
import os
import numpy as np
import time

from common.configurations.settings import Settings

gin.parse_config_file(f'common/configurations/settings.gin')

from common.base_simulation import BaseSimulation
from controllers.random_controller import RandomController


def parse_configuration(base_simulation_configuration: int, controller_type: str, controller_config: int) -> None:
    """
    :param base_simulation_configuration: id of the base_configuration.
        This field should match a number of configuration in common/configurations/base_configuration{i}.gin
    :param controller_type: name of a controller class in controllers/
    :param controller_config: id of the controller configuration
        This field should match a number of configuration in controllers/configurations/controller_type/base_configuration{i}.gin
    :return: None

    Create and arse the system and controller configuration using gin.
    A temporary file gathering all the configurations is created, parsed and then deleted.
    The parameters may be saved at the end of the simulation using gin.
    """

    with open(f'./common/configurations/base_simulation_configuration{base_simulation_configuration}.gin', "r") as file:
        original_config = file.readlines()

    modified_config = f"include './controllers/configurations/{controller_type}/config{controller_config}.gin'\n"
    for line in original_config:
        if line.startswith("BaseSimulation.controller_class =") and controller_type is not None:
            modified_config += f"BaseSimulation.controller_class = @{controller_type}\n"
        else:
            modified_config += line

    # Save the modified configuration to a temporary file, parse config does not handle include statements
    nanoseconds = time.time_ns()
    temp_filename = f"temp_config_{nanoseconds}.gin"  # create a unique file for parallel jobs
    with open(temp_filename, "w") as temp_file:
        temp_file.write(modified_config)
    gin.parse_config_file(temp_filename)
    os.remove(temp_filename)


def main(
    base_simulation_configuration_id: int,
    controller_configuration_id: int,
    controller_type: str = 'RandomController'
) -> None:
    """
    :param base_simulation_configuration_id: id of the base_configuration.
        This field should match a number of configuration in common/configurations/base_configuration{i}.gin
    :param controller_type: name of a controller class in controllers/
    :param controller_configuration_id: id of the controller configuration
        This field should match a number of configuration in controllers/configurations/controller_type/base_configuration{i}.gin
    :return: None

    Run a base simulation
    """

    parse_configuration(
        base_simulation_configuration=base_simulation_configuration_id,
        controller_type=controller_type,
        controller_config=controller_configuration_id
    )
    settings = Settings()
    if settings.simulation_seed is not None:
        print('Fixing simulation seed')
        np.random.seed(settings.simulation_seed)
    else:
        print('Warning: the simulation seed is not fixed')
    base_simulation = BaseSimulation()
    base_simulation.run()
