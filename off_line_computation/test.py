import gin
import jax.numpy as jnp

from state_space_models.configurations.utils import modify_and_parse_test_config
from off_line_computation.utils import load_norm_constants, load_data, prepare_batches, get_processed_state_indexes, \
    compute_score
from common.utils import save_config_file, save_results
from typing import List


def test_models(
    model_type: str,
    building: str,
    simulation_id: int,
    zone_id: int,
    test_config: int,
    base_config_id: int,
    configs: List[int],
) -> None:
    """
    :param model_type: name of the model to test
    :param building: name of the building data are collected on. Should match the name of the folder in /data
    :param simulation_id: id of the simulation data were collected on
    :param zone_id: id of the zone data were collected on
    :param test_config: id of the test_config file
    :param base_config_id: number of the base train test configuration to parse
    :param configs: list of ids of model configurations to test. Check in configuration/models the available configs.

    test the models with the given configurations. Store the weights of the models in the results folder.
    """

    for i in configs:
        test_model_on_single_config(
            model_type=model_type,
            building=building,
            simulation_id=simulation_id,
            zone_id=zone_id,
            test_config=test_config,
            base_config_id=base_config_id,
            model_config_id=i
        )


def test_model_on_single_config(
    model_type: str,
    building: str,
    zone_id: int,
    simulation_id: int,
    test_config: int,
    model_config_id: int,
    base_config_id: int,
    verbose: bool = True
) -> None:
    """
    :param model_type: name of the model, should match one of the models class name
    :param building: name of the building data are collected on. Should match the name of the folder in /data
    :param simulation_id: id of the simulation data were collected on
    :param zone_id: id of the zone data were collected on
    :param test_config: number of the base test configuration to parse
    :param base_config_id: number of the base test configuration to parse
    :param model_config_id: number of the model configuration to use for test
    :param verbose: whether to display logs while test

    Launch a model test
    """

    if verbose:
        print(f'------------ test configuration {model_config_id} ------------')

    modify_and_parse_test_config(
        base_config_path=f'./state_space_models/configurations/train_test/test_config{test_config}.gin',
        model_config_id=model_config_id,
        base_config_id=base_config_id,
        model_type=model_type
    )
    test(
        test_config=test_config,
        building=building,
        zone_id=zone_id,
        simulation_id=simulation_id,
        model_config_id=model_config_id,
        validation_data=True
    )
    test(
        test_config=test_config,
        building=building,
        zone_id=zone_id,
        simulation_id=simulation_id,
        model_config_id=model_config_id,
        validation_data=False
    )


@gin.configurable
def test(
    test_config: int,
    building: str,
    zone_id: int,
    simulation_id: int,
    model_config_id: int,
    model,
    batch_size: int,
    batch_length: int,
    validation_data: bool = False,
    seeds: List[int] = None
) -> None:
    """
    :param test_config: id of the test configuration
    :param building: name of the building data are collected on. Should match the name of the folder in /data
    :param simulation_id: id of the simulation data were collected on
    :param zone_id: id of the zone data were collected on
    :param model_config_id: id of the model configuration used for training
    :param model: model class
    :param batch_size:
    :param batch_length:
    :param validation_data: wheter to perform the test using the validation or test dataset
    :param seeds: list of seeds to use for testing. If several seeds are specified, tests are repeated for each seed.
    :return:
    """

    model_type = model.__name__
    data_path = f'common/results/{building}/simulation_{simulation_id}/processed_data'
    model_dir = f'common/results/{building}/simulation_{simulation_id}/models/{model_type}/zone{zone_id}/train_test_{test_config}_model_{model_config_id}'

    data_type = 'validation' if validation_data else 'test'

    print(f"--- {data_type} testing ---")
    print(f"batch_size: {batch_size}")

    if seeds is not None:
        for seed in seeds:
            print(f' --- Testing with seed {seed} ---')
            mean, std = load_norm_constants(zone_id, data_path)
            state_indexes = get_processed_state_indexes(data_path)
            norm_test_data = load_data(zone_id, data_path, data_type=data_type)
            test_data = prepare_batches(jnp.array(norm_test_data), batch_length)
            model = model(
                seed=seed,
                mean=mean,
                std=std,
                state_indexes=state_indexes,
                model_path=model_dir
            )
            results = model.test_model(
                test_data=test_data,
                batch_size=batch_size,
            )
            compute_score(results)
            config_str = gin.operative_config_str()
            save_config_file(config_str, model_dir + f'/{data_type}_config/')
            save_results(results, zone_id, model_dir, seed=seed, validation=validation_data)
    else:
        mean, std = load_norm_constants(zone_id, data_path)
        state_indexes = get_processed_state_indexes(data_path)
        norm_test_data = load_data(zone_id, data_path, data_type=data_type)
        test_data = prepare_batches(jnp.array(norm_test_data), batch_length)
        model = model(
            zone_id=zone_id,
            mean=mean,
            std=std,
            state_indexes=state_indexes,
            model_path=model_dir
        )
        results = model.test_model(
            test_data=test_data,
            batch_size=batch_size,
        )
        compute_score(results)
        config_str = gin.operative_config_str()
        save_config_file(config_str, model_dir + f'/{data_type}_config/')
        save_results(results, zone_id, model_dir, validation=validation_data)
