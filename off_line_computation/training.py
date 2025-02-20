import gin
import os
import jax.numpy as jnp

from off_line_computation.utils import load_norm_constants, get_processed_state_indexes, load_data, prepare_batches
from common.utils import save_config_file
from state_space_models.configurations.utils import modify_and_parse_training_config
from typing import List


def train_models(
    model_type: str,
    building: str,
    simulation_id: int,
    zone_id: int,
    training_config: int,
    base_config_id: int,
    configs: List[int],
) -> None:
    """
    :param model_type: name of the model to train
    :param building: name of the building data are collected on. Should match the name of the folder in /data
    :param simulation_id: id of the simulation data were collected on
    :param zone_id: id of the zone data were collected on
    :param training_config: id of the training_config file
    :param base_config_id: number of the base train test configuration to parse
    :param configs: list of ids of model configurations to train. Check in configuration/models the available configs.

    Train the models with the given configurations. Store the weights of the models in the results folder.
    """

    for i in configs:
        train_model_on_single_config(
            model_type=model_type,
            building=building,
            simulation_id=simulation_id,
            zone_id=zone_id,
            training_config=training_config,
            base_config_id=base_config_id,
            model_config_id=i
        )


def train_model_on_single_config(
    model_type: str,
    building: str,
    simulation_id: int,
    zone_id: int,
    training_config: int,
    model_config_id: int,
    base_config_id: int,
    verbose: bool = True
) -> None:
    """
    :param model_type: name of the model, should match one of the models class name
    :param building: name of the building data are collected on. Should match the name of the folder in /data
    :param simulation_id: id of the simulation data were collected on
    :param zone_id: id of the zone data were collected on
    :param training_config: number of the base train test configuration to parse
    :param base_config_id: number of the base train test configuration to parse
    :param model_config_id: number of the model configuration to use for training
    :param verbose: whether to display logs while training

    Launch a model training
    """

    if verbose:
        print(f'------------ training configuration {model_config_id} ------------')

    modify_and_parse_training_config(
        base_config_path=f'./state_space_models/configurations/train_test/training_config{training_config}.gin',
        model_config_id=model_config_id,
        base_config_id=base_config_id,
        model_type=model_type
    )
    training(
        training_config_id=training_config,
        building=building,
        zone_id=zone_id,
        simulation_id=simulation_id,
        model_config_id=model_config_id
    )


@gin.configurable
def training(
    training_config_id: int,
    building: str,
    simulation_id: int,
    zone_id: int,
    model,
    model_config_id: int,
    batch_length: int,
    training_batch_size: int,
    validation_batch_size: int,
    loss_temperature_weight: float,
    training_steps: int,
    validation_every: int,
    validation: bool,
    early_stopping: bool
) -> None:
    """
    :param training_config_id: id of the training configuration to use
    :param building: name of the building data are collected on. Should match the name of the folder in /data
    :param simulation_id: id of the simulation data were collected on
    :param zone_id: id of the zone data were collected on
    :param model: class of model define in the gin config file
    :param model_config_id: id of the configuration to use for the training
    :param batch_length:
    :param training_batch_size:
    :param validation_batch_size:
    :param loss_temperature_weight: Parameter used to balance the importance of the temperature prediction in the loss
    :param training_steps: number of training steps
    :param validation_every: number of training steps between two validations
    :param early_stopping: whether to apply early stopping
    :param validation: whether to do validation during training
    :return:
    """
    print("--- def training() ---")
    print(f"model: {model}")
    print(f"batch_length: {batch_length}")

    model_type = model.__name__
    base_path = f'common/results/{building}/simulation_{simulation_id}'
    data_path = base_path + f'/processed_data'
    model_dir = base_path + f'/models/{model_type}/zone{zone_id}/train_test_{training_config_id}_model_{model_config_id}'

    # save config file
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    config_str = gin.operative_config_str()
    save_config_file(config_str, model_dir + f'/training_config/')
    # Training data
    mean, std = load_norm_constants(zone_id, data_path)
    state_indexes = get_processed_state_indexes(data_path)
    norm_training_data = load_data(zone_id, data_path, 'training')
    print(f"norm_training_data.shape: {norm_training_data.shape}")

    training_data = prepare_batches(jnp.array(norm_training_data), batch_length)

    # Validation data
    if validation:
        norm_validation_data = load_data(zone_id, data_path, 'validation')
        print(f"norm_validation_data.shape: {norm_validation_data.shape}")

        validation_data = prepare_batches(norm_validation_data, batch_length)
        del norm_validation_data, norm_training_data
    else:
        validation_data = None

    model = model(
        zone_id=zone_id,
        mean=mean,
        std=std,
        state_indexes=state_indexes
    )

    print(f"training_data.shape: {training_data.shape}")
    print(f"validation_data.shape: {validation_data.shape}")

    model.train_model(
        training_data=training_data,
        validation_data=validation_data,
        validation_every=validation_every,
        training_batch_size=training_batch_size,
        validation_batch_size=validation_batch_size,
        training_steps=training_steps,
        loss_temperature_weight=loss_temperature_weight,
        early_stopping=early_stopping,
        model_path=model_dir
    )

    model.save_model(path=model_dir, final=True)
    model.save_training_logs(path=model_dir)
