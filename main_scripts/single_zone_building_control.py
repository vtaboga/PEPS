import time
import os
import gin
import numpy as np

from jax import config
from common.system_single_zone_building import SingleZoneBuildingSimulation
from common.configurations.settings import Settings
from controllers.shooting_controller import ShootingController
from controllers.rule_based_controller import RuleBasedController

gin.parse_config_file(f'common/configurations/settings.gin')


def parse_configuration(
    simulation_config: int,
    controller_type: str,
    controller_config: int,
    model_type: str,
    model_config: int,
    base_model_config: int
) -> None:
    """
    :param simulation_config:
    :param controller_type:
    :param controller_config:
    :param model_type:
    :param model_config:
    :param base_model_config:
    :return:
    """

    modified_config = f"include './state_space_models/configurations/train_test/base_config{base_model_config}.gin' \n"
    modified_config += f"include './state_space_models/configurations/models/{model_type}/config{model_config}.gin'\n"
    modified_config += f"include './controllers/configurations/{controller_type}/config{controller_config}.gin'\n"

    # Add controller configuration and replace the model type by the one specified in the function's parameters
    with open(f'./controllers/configurations/{controller_type}/config{controller_config}.gin', "r") as file:
        controller_config = file.readlines()
    for line in controller_config:
        if line.startswith(f"{controller_type}.prediction_model ="):
            modified_config += f"{controller_type}.prediction_model = @{model_type}\n"
        else:
            modified_config += line
    # Add simulation configuration and replace the controller type by the one specified in the function's parameters
    with open(f'./common/configurations/single_zone_configuration{simulation_config}.gin', "r") as file:
        original_config = file.readlines()
    for line in original_config:
        if line.startswith("SingleZoneBuildingSimulation.controller_class =") and controller_type is not None:
            modified_config += f"SingleZoneBuildingSimulation.controller_class = @{controller_type}\n"
        else:
            modified_config += line

    # Save the modified configuration to a temporary file and parse it
    nanoseconds = time.time_ns()
    temp_filename = f"temp_config_{nanoseconds}.gin"  # create a unique file for parallel jobs
    with open(temp_filename, "w") as temp_file:
        temp_file.write(modified_config)
    gin.parse_config_file(temp_filename)
    # gin.parse_config_file("temp_config_1738836566854336000.gin")
    os.remove(temp_filename)


def main(
    simulation_configuration: int,
    controller_type: str,
    controller_config: int,
    model_type: str,
    model_config: int,
    training_config: int,
    base_model_config: int
) -> None:

    parse_configuration(
        simulation_configuration,
        controller_type,
        controller_config,
        model_type,
        model_config,
        base_model_config
    )
    settings = Settings()
    config.update("jax_enable_x64", settings.jax_enable_x64)

    if settings.simulation_seed is not None:
        print('Fixing simulation seed')
        np.random.seed(settings.simulation_seed)
    else:
        print('Warning: the simulation seed is not fixed')
    building_type = gin.query_parameter('%BUILDING_TYPE')
    model_path = f'./common/results/{building_type}/simulation_{simulation_configuration}/models/{model_type}/zone{0}/train_test_{training_config}_model_{model_config}'
    simulation = SingleZoneBuildingSimulation(model_path=model_path)
    simulation.run()
