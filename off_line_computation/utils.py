import fnmatch
import jax.numpy as jnp
import numpy as np
import pandas as pd
import re
import itertools
import copy
import json
import os

from typing import List, Dict


def save_norm_constants(zone_id: int, mean: np.array, std: np.array, processed_data_dir: str) -> None:
    """
    :param zone_id: id of the zone data were collected in
    :param mean: array of mean values of the features
    :param std: array of std values of the features
    :param processed_data_dir: path to the directory to save the normalization constants
        This should be the directory were the processed data have been saved
    :return:
    """
    path_dir = processed_data_dir + f'/zone{zone_id}'
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)
    with open(path_dir + f"/mean.json", 'w') as f:
        json.dump(mean.tolist(), f)
    with open(path_dir + f"/std.json", 'w') as f:
        json.dump(std.tolist(), f)


def load_norm_constants(zone_id: int, processed_data_dir: str) -> (np.array, np.array):
    """
    :param zone_id: id of the zone data were collected in
    :param processed_data_dir: path to the directory where normalization constants have been saved
    :return: arrays of mean and std for the features
    """
    path_dir = processed_data_dir + f'/zone{zone_id}'
    with open(path_dir + f"/mean.json", 'r') as f:
        mean = np.array(json.load(f))
    with open(path_dir + f"/std.json", 'r') as f:
        std = np.array(json.load(f))
    return mean, std


def load_data(zone_id: int, processed_data_dir: str, data_type: str) -> np.array:
    """
    :param zone_id: id of the zone data were collected in
    :param processed_data_dir: path to the directory where the processed data of each zone have been saved
    :param data_type: training, validation or test data
    :return: array containing the data set (training, validation or test)
    """
    data = np.loadtxt(processed_data_dir + f'/zone{zone_id}/{data_type}.csv', delimiter=',')
    return data


def prepare_batches(data: jnp.array, batch_length: int) -> jnp.array:
    """
    :param data: normalized data set (training, validation or test)
    :param batch_length: length of a data batch
    :return: data set with an extra dimension corresponding to the batch size.
    """

    n_batches = data.shape[0] // batch_length
    batched_data = []
    for i in range(n_batches):
        batched_data.append(data[i * batch_length:(i + 1) * batch_length, :])

    return jnp.array(batched_data)


def get_processed_state_indexes(processed_data_dir: str) -> Dict:
    with open(processed_data_dir + f'/processed_state_indexes.json', 'r') as f:
        return json.load(f)


def mean_absolute_percentage_error(targets: np.array, predictions: np.array):
    return np.mean(np.abs((targets - predictions) / targets)) * 100


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def group_configs(grid: Dict, start_number: int = 1) -> Dict:
    """
    :param grid: dictionary used for a grid search
    :param start_number: number of the first configuration to consider
    :return: The id of the configuration files grouped by configurations (for all the seeds)
    """
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    groups = {}
    for i, exp in enumerate(experiments):
        exp_copy = copy.deepcopy(exp)
        if 'seed' in exp_copy:
            del exp_copy['seed']
        key = tuple(sorted(exp_copy.items()))
        if key not in groups:
            groups[key] = []
        groups[key].append(i + start_number)

    return groups


def parse_results(
    controller_type: str,
    train_test_config: int,
    energyplus_timestep_in_hour: int,
    experiment_number: int = 1,
    building_type: str = 'house',
    config_range: tuple = None,
    grid: dict = None
) -> (pd.DataFrame, pd.DataFrame):
    """
        :param grid: the dictionary used for the grid search
        :param train_test_config: the train_test_config number used for the training / testing
        :param controller_type: name of the controller class used
        :param building_type: name of the building on which data were collected
        :param experiment_number: number of the configuration used to collect data
        usually 1 for 5 minutes time steps and 2 for 1 minute time steps (irregular data)
        :param config_range: tuple (start config, end config) to specify the config range to parse
        useful to parse configuration for a grid search with config number from 61 to 75 for instance.

        Data frames or results ordered by best power accuracy and temperature accuracy respectively.
        """

    mape_results = []
    results_dir = f'local_controllers/saved_models/{building_type}/{controller_type}'
    all_items = os.listdir(results_dir)
    pattern = f'experiment_{experiment_number}_config_{train_test_config}_' + r'(\d+)'
    config_folders = [f for f in all_items if os.path.isdir(os.path.join(results_dir, f)) and re.match(pattern, f)]
    configurations = list(itertools.product(*[v for k, v in grid.items()]))

    for folder_name in config_folders:
        match = re.match(pattern, folder_name)
        if match is None:
            continue
        config_number = int(match.group(1))

        # check if config number is within specified range
        if config_range and (config_number < config_range[0] or config_number > config_range[1]):
            continue

        csv_path = os.path.join(results_dir, folder_name, 'zone0/test_results.csv')
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            df['power_targets'] = df['power_targets'] * energyplus_timestep_in_hour
            df['power_predictions'] = df['power_predictions'] * energyplus_timestep_in_hour
            power_mape = mean_absolute_percentage_error(df['power_targets'], df['power_predictions'])
            temperature_mape = mean_absolute_percentage_error(
                df['temperature_targets'],
                df['temperature_predictions']
            )
            params = configurations[config_number - 1]
            mape_results.append((config_number,) + tuple(params) + (power_mape, temperature_mape))
        else:
            print(f"No results csv file found in {folder_name}. Skipping this folder.")

    columns = ['Config Number'] + [key for key in grid.keys()] + ['Power MAPE', 'Temperature MAPE']
    df_power = pd.DataFrame(sorted(mape_results, key=lambda x: x[-2]), columns=columns)
    df_temperature = pd.DataFrame(sorted(mape_results, key=lambda x: x[-1]), columns=columns)

    return df_power, df_temperature


def parse_results_per_seed(
    grid: dict,
    train_test_config: int,
    energyplus_timestep_in_hour: int,
    model_type: str,
    start_config_number: int,
    validation: bool,
    building_type: str = 'house',
    zone_number: int = 0,
    experiment_number: int = 1,
):
    """
        :param grid: the dictionary used for the grid search
        :param train_test_config: the train_test_config number used for the training / testing
        :param model_type: name of the controller class used
        :param start_config_number: id of the first config to parse.
        :param building_type: name of the building on which data were collected
        :param experiment_number: number of the configuration used to collect data
                                  usually 1 for 5 minutes time steps and 2 for 1 minute time steps (irrgular data)
        :param disp: to display parsing infomration

        The function parse the results of a grid search test files. The score (MAPE) is computed and stored in a df

        :return: a data frame gathering the score for each config and each seed.
        """
    results_dir = f'common/results/{building_type}/simulation_{experiment_number}/models/{model_type}/zone{zone_number}'
    configurations = group_configs(grid, start_config_number)
    mape_results = pd.DataFrame()

    for parameters, config_indices in configurations.items():
        configuration_number = config_indices[0]
        for i in config_indices:
            try:
                folder_name = f'train_test_{train_test_config}_model_{i}'
                results = get_test_scores(
                    os.path.join(results_dir, folder_name),
                    config=i,
                    validation=validation,
                    energyplus_timestep_in_hour=energyplus_timestep_in_hour
                )
                if results.shape[0] == 0:
                    print(f"No csv file found in {folder_name}. Skipping this folder.")
                else:
                    parameters_dict = dict(parameters)
                    for parameter, value in parameters_dict.items():
                        results[parameter] = value
                    results['configuration_number'] = configuration_number
                    mape_results = pd.concat([mape_results, results], axis=0)
            except FileNotFoundError:
                print(f"No csv file found for config {i}. Skipping this folder.")

    return mape_results


def get_test_scores(results_path: str, config: int, validation: bool, energyplus_timestep_in_hour: int) -> pd.DataFrame:
    """
    :param results_path: path to the directory where test cv files are stored
    :param config: configuration number (for the local controller)
    :return: a Data Frame containing the power and temperature mape for each test seed
    """

    file_name = f'validation' if validation else f'test'

    def extract_number(filename):
        match = re.search(rf'{filename}_results_seed_(\d+).csv', filename)
        if match:
            return int(match.group(1))
        return 0

    files = os.listdir(results_path)
    results_files = fnmatch.filter(files, f'{file_name}_results_seed_*.csv')
    if len(results_files) == 0:
        results_files = [f'{file_name}_results.csv']
    else:
        results_files = sorted(results_files, key=extract_number)

    power_mapes = []
    temperature_mapes = []
    power_rmses = []
    temperature_rmses = []
    power_maes = []
    temperature_maes = []
    seeds = []

    for seed, res in enumerate(results_files):
        df = pd.read_csv(os.path.join(results_path, res))

        df['power_targets'] = df['hvac_power_targets'] * energyplus_timestep_in_hour
        df['power_predictions'] = df['hvac_power_predictions'] * energyplus_timestep_in_hour

        # MAPE calculations
        power_mape = mean_absolute_percentage_error(df['power_targets'], df['power_predictions'])
        temperature_mape = mean_absolute_percentage_error(df['temperature_targets'], df['temperature_predictions'])

        # RMSE calculations
        power_rmse = root_mean_square_error(df['power_targets'], df['power_predictions'])
        temperature_rmse = root_mean_square_error(df['temperature_targets'], df['temperature_predictions'])

        # MAE calculations
        power_mae = mean_absolute_error(df['power_targets'], df['power_predictions'])
        temperature_mae = mean_absolute_error(df['temperature_targets'], df['temperature_predictions'])

        power_mapes.append(power_mape)
        temperature_mapes.append(temperature_mape)
        power_rmses.append(power_rmse)
        temperature_rmses.append(temperature_rmse)
        power_maes.append(power_mae)
        temperature_maes.append(temperature_mae)
        seeds.append(seed + 1)

    results = pd.DataFrame(
        {
            'test_seed': seeds,
            'temperature_mape': temperature_mapes,
            'power_mape': power_mapes,
            'temperature_rmse': temperature_rmses,
            'power_rmse': power_rmses,
            'temperature_mae': temperature_maes,
            'power_mae': power_maes
        }
    )
    results['config'] = [config] * results.shape[0]
    return results


def compute_statistics(results: pd.DataFrame, sort_on: str = 'power') -> pd.DataFrame:
    """
    :param results: results from parse_results_per_seed
    :param sort_on: sort the data frame for temperature or power
    :return: mean and std mape scores for each config
    """

    exclude_columns = {'seed', 'power_mape', 'temperature_mape', 'configuration_number', 'config'}
    param_columns = [col for col in results.columns if col not in exclude_columns]
    agg_dict = {
        'power_mape': ['mean', 'std', 'min', 'max'],
        'temperature_mape': ['mean', 'std', 'min', 'max'],
    }

    agg_dict.update(
        {col: ['first'] for col in param_columns}
    )  # keep first value of the param columns (same across the group by)
    grouped = results.groupby('configuration_number')
    statistics = grouped.agg(agg_dict)
    statistics.columns = ['_'.join(col) for col in statistics.columns.values]
    if sort_on == 'power':
        statistics = statistics.sort_values('power_mape_mean', ascending=True)
    elif sort_on == 'temperature':
        statistics = statistics.sort_values('temperature_mape_mean', ascending=True)
    else:
        raise ValueError(f'Field {sort_on} unknown.')

    return statistics


def load_results(
    controller_type: str,
    building_type: str,
    configuration: int,
    train_test_config: int,
    experiment: int,
    energyplus_timestep_in_hour: int,
    zone_number: int = None,
    seed: int = None,
    one_test_dataset: bool = False,
    start_date: str = '01-01',
    end_date: str = '03-15'
) -> pd.DataFrame:
    """
    :param controller_type: name of the controller class used
    :param building_type: name of the building data were collected on
    :param configuration: number of the configuration file (for the controller) to load
    :param train_test_config: number of the train_test_config used to test the model
    :param experiment: number of the configuration used to collect data
                              usually 1 for 5 minutes time steps and 2 for 1 minute time steps (irrgular data)
    :param zone_number: number of the zone the controller is acting one (usually None, the house model has one zone)
    :param seed: seed used for the test

    Load the results, containing the power and temperature predictions and true values for the test period.
    Tests data range from Jan 1 to March 15. Two tests periods are concatenated.

    """

    sim_timestep = int(60 / energyplus_timestep_in_hour)

    test_file = 'test_results.csv' if seed is None else f'test_results_seed_{seed}.csv'

    if train_test_config is None:
        results_dir = f'local_controllers/saved_models/{building_type}/{controller_type}/experiment_{experiment}_config{configuration}'
    else:
        results_dir = f'local_controllers/saved_models/{building_type}/{controller_type}/experiment_{experiment}_config_{train_test_config}_{configuration}'
    if zone_number is None:
        results_path = results_dir + f'/zone0/{test_file}'
    else:
        results_path = results_dir + f'/zone{zone_number}/{test_file}'
    results = pd.read_csv(results_path)
    if one_test_dataset:
        date_range = pd.date_range(
            start=f'2021-{start_date} 01:00:00',
            end=f'2021-{end_date} 23:55:00',
            freq=f'{sim_timestep}min'
        )
        date_range = date_range[:results.shape[0]]
        results = results.iloc[:len(date_range)].set_index(date_range)
    else:
        date_range_1 = pd.date_range(
            start=f'2018-{start_date} 01:00:00',
            end=f'2018-{end_date} 23:55:00',
            freq=f'{sim_timestep}min'
        )
        date_range_2 = pd.date_range(
            start=f'2021-{start_date} 01:00:00',
            end=f'2021-{end_date} 23:55:00',
            freq=f'{sim_timestep}min'
        )
        combined_date_range = pd.concat([pd.Series(date_range_1), pd.Series(date_range_2)]).reset_index(drop=True)
        # Tests are sometimes made twice over the test data. Remove duplicates.
        results = results.iloc[:len(combined_date_range)].set_index(combined_date_range)

    results['power_predictions'] = results['power_predictions'] * energyplus_timestep_in_hour
    results['power_targets'] = results['power_targets'] * energyplus_timestep_in_hour

    return results


def compute_score(results: Dict) -> None:
    """
    :param results: test results containing predictions and true values of the state components
    :return:
    Compute and display the scores
    """
    power_targets = results['hvac_power_targets']
    power_preds = results['hvac_power_predictions']
    temperature_targets = results['temperature_targets']
    temperature_preds = results['temperature_predictions']
    mape = np.mean(np.abs((power_targets - power_preds) / power_targets)) * 100
    mae = np.mean(np.abs((power_targets - power_preds)))
    rmse = jnp.sqrt(jnp.mean((power_preds - power_targets) ** 2))
    print('--------------- HVAC energy predictions ----------------')
    print(f'MAPE : {jnp.round(mape, 4)} %')
    print(f'MAE : {np.round(mae, 4)} kw')
    print(f'RMSE : {np.round(rmse, 4)} kw')
    mape = np.mean(np.abs((temperature_targets - temperature_preds) / temperature_targets)) * 100
    mae = np.mean(np.abs((temperature_targets - temperature_preds)))
    rmse = jnp.sqrt(jnp.mean((temperature_preds - temperature_targets) ** 2))
    print('--------------- Temperature predictions ----------------')
    print(f'MAPE : {jnp.round(mape, 4)} %')
    print(f'MAE : {jnp.round(mae, 4)} deg C')
    print(f'RMSE : {jnp.round(rmse, 4)} deg C')
