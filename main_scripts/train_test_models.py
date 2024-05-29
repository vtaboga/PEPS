import gin

from off_line_computation.training import train_models
from off_line_computation.test import test_models
from off_line_computation.processing import process_data
from common.utils import save_config_file
from state_space_models.configurations.utils import create_configs
from typing import List
from state_space_models.state_space_model import SSM
from state_space_models.structured_state_space_model import StructuredSSM
from state_space_models.continuous_encoder_state_space_model import CESSM
from state_space_models.latent_neural_ode import LatentNeuralODE
from state_space_models.neural_ode import NeuralODE
from state_space_models.structured_latent_neural_ode import StructuredLatentNeuralODE
from state_space_models.continuous_encoder_latent_neural_ode import CELatentNeuralODE
from state_space_models.controlled_differential_equation import CDE
from state_space_models.recurrent_state_space_model import RSSM


def main(
    training: bool,
    test: bool,
    create_model_configs: bool,
    start_config_number: int,
    processing_data: bool,
    simulation_id: int,
    processing_data_config: int,
    model_type: str,
    building: str,
    zone_id: int,
    training_config: int,
    test_config: int,
    base_config: int,
    model_configs: List[int]
) -> None:
    """
    :param training: Whether to train the model
    :param test: Whether to test the model
    :param create_model_configs: Whether to create model configuration files using pre-defined grid search
    :param start_config_number: If configuration files are created, id of the first configuration file.
          If configuration files with the same id exists, they will be replaced.
    :param simulation_id: Id of the simulation training and test data were collected on
    :param processing_data: Whether to process the data
    :param processing_data_config: If processing data, id of the configuration to use
    :param model_type: Type of prediction model to use.
    :param building: Name of the building data are collected on. Should match the name of the folder in /data
    :param zone_id: Id of the zone data were collected on
    :param training_config: If training, id of the training configuration to use
    :param test_config: If testing, id of the test configuration to use
    :param base_config: Id of the base training / test configuration
    :param model_configs: Ids of the model configuration to use. If several ids are specified, the training and/or test
          is repeated for each id.
    :return: None

    Call the model's training or test method.
    If specified, process the data and create configurations for the models.
    """

    if processing_data:

        gin.parse_config_file(
            f'./state_space_models/configurations/train_test/processing_config{processing_data_config}.gin'
        )
        process_data()
        config_str = gin.operative_config_str()
        save_config_file(
            config_str,
            f'common/results/{building}/simulation_{simulation_id}/processed_data/zone{zone_id}/',
            base_filename='processing_config'
        )

    if create_model_configs:
        create_configs(model_type=model_type, start_number=start_config_number)

    if training:
        train_models(
            model_type=model_type,
            building=building,
            zone_id=zone_id,
            training_config=training_config,
            base_config_id=base_config,
            configs=model_configs,
            simulation_id=simulation_id
        )
    if test:
        test_models(
            model_type=model_type,
            building=building,
            simulation_id=simulation_id,
            zone_id=zone_id,
            test_config=test_config,
            base_config_id=base_config,
            configs=model_configs
        )
