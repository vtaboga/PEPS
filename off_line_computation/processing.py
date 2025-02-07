import gin
import os
import json
import pandas as pd
import numpy as np

from typing import List, Dict
from off_line_computation.utils import save_norm_constants, load_norm_constants


@gin.configurable
def process_data(
    building_type: str,
    simulation_id: int,
    simulation_results_dir: str,
    number_of_zones: int,
    n_lags: int,
    heating_season: bool,
    setpoint_change: bool,
    include_action_lags: bool,
    energyplus_timesteps_in_hour: int,
    prediction_horizon: int,
    resample_training_data: bool,
    resample_test_data: bool,
    training_data_indexes: List[int],
    validation_data_indexes: List[int],
    test_data_indexes: List[int]
) -> None:
    """
    :param building_type: name of the building used for the simulation.
    :param simulation_id: id of the simulation data were collected on
    :param simulation_results_dir: path to the results the data collected from EnergyPlus
    :param number_of_zones: number of zones in the building
    :param n_lags: number of past observations to include to the lags
    :param heating_season: whether data were collected during winter (heating season)
    :param setpoint_change: whether to include the difference between two consecutive temperature setpoints in the data
    :param include_action_lags: whether to include lags of actions in the observations
    :param energyplus_timesteps_in_hour: number of EnergyPlus timesteps in one hour of simulation
    :param prediction_horizon: prediction horizon of the models trained with these data
    :param resample_training_data: whether to resample training data around temperature setpoint changes
    :param resample_test_data: whether to resample test data around temperature setpoint changes
    :param training_data_indexes: file numbers to use for training
    :param validation_data_indexes: file numbers to use for validation (may be an empty list)
    :param test_data_indexes: file numbers to use for test (may be an empty list)
    :return: None

    Preprocess the simulation data if the data have not already been preprocessed
    Standardize the data
    Save the standardization constants
    Create training validation and test sets
    """

    processed_data_dir = simulation_results_dir + f'{building_type}/simulation_{simulation_id}/processed_data/'
    simulation_results_dir += f'{building_type}/simulation_{simulation_id}/data'

    if not os.path.isdir(processed_data_dir):
        preprocessing(
            number_of_zones=number_of_zones,
            n_lags=n_lags,
            heating_season=heating_season,
            setpoint_change=setpoint_change,
            simulation_results_dir=simulation_results_dir,
            processed_data_dir=processed_data_dir,
            include_action_lags=include_action_lags,
            energyplus_timesteps_in_hour=energyplus_timesteps_in_hour
        )
        # standardize data (except cyclical features)
        with open(processed_data_dir + 'processed_state_indexes.json', 'r') as f:
            processed_state_indexes = json.load(f)

        standardize_data(
            number_of_zones=number_of_zones,
            processed_data_dir=processed_data_dir,
            state_indexes=processed_state_indexes,
            data_type='training',
            setpoint_change=setpoint_change,
            data_indexes=training_data_indexes,
            resample_data=resample_training_data,
            prediction_horizon=prediction_horizon
        )

        if len(validation_data_indexes) > 0:
            standardize_data(
                number_of_zones=number_of_zones,
                processed_data_dir=processed_data_dir,
                state_indexes=processed_state_indexes,
                data_type='validation',
                setpoint_change=setpoint_change,
                data_indexes=validation_data_indexes,
                resample_data=resample_test_data,
                prediction_horizon=prediction_horizon
            )

        if len(test_data_indexes) > 0:
            standardize_data(
                number_of_zones=number_of_zones,
                processed_data_dir=processed_data_dir,
                state_indexes=processed_state_indexes,
                data_type='test',
                setpoint_change=setpoint_change,
                data_indexes=test_data_indexes,
                resample_data=resample_test_data,
                prediction_horizon=prediction_horizon
            )
    else:
        print('The processing of the simulation data has already been done.')


def preprocessing(
    simulation_results_dir: str,
    processed_data_dir: str,
    number_of_zones: int,
    n_lags: int,
    heating_season: bool,
    setpoint_change: bool,
    include_action_lags: bool,
    energyplus_timesteps_in_hour: int
) -> None:
    """
    :param number_of_zones: number of zones in the building
    :param n_lags: number of past observations to include to the lags
    :param heating_season: whether data were collected during winter (heating season)
    :param setpoint_change: whether to include the difference between two consecutive temperature setpoints in the data
    :param simulation_results_dir: path to the results the data collected from EnergyPlus
    :param processed_data_dir: path to the directory to store the processed data
    :param include_action_lags: whether to include lags of actions in the observations
    :param energyplus_timesteps_in_hour: number of EnergyPlus timesteps in one hour of simulation
    :return: None

    Preprocess the simulation results.
    A row of preprocessed data is composed of the lags of observations as well as the target:
    [P(t), T(t), lags P, lags T, weather(t), set point(t)].
    A prediction model would take as input [lags P, lags T, weather(t), setpoint(t)] to predict P(t), T(t).
    """
    files_number = len(os.listdir(simulation_results_dir))
    simulation_time_step = 60 / energyplus_timesteps_in_hour  # simulation_time_step in minutes

    for zone_number in range(number_of_zones):
        # load all the data files from the simulations (.pickle files)
        data = []
        for i in range(files_number):
            file_path = os.path.join(simulation_results_dir, f'{i}.pickle')
            data.append(
                pd.DataFrame(
                    pd.read_pickle(file_path)
                )
            )
        # parse each data file
        data_files, state_indexes = parse_base_simulation_results(data, zone_number, heating_season)
        for i, data_file in enumerate(data_files):
            # preprocess the data files to add features
            df = add_features(
                data_file=data_file,
                state_indexes=state_indexes,
                setpoint_change=setpoint_change,
                n_lags=n_lags,
                simulation_time_step=simulation_time_step,
                include_action_lags=include_action_lags
            )
            # save results
            if not os.path.exists(processed_data_dir + f'zone{zone_number}'):
                os.makedirs(processed_data_dir + f'zone{zone_number}')
            df.to_csv(processed_data_dir + f'zone{zone_number}/processed_data{i}.csv', index=False)
            if not os.path.exists(processed_data_dir + f'processed_state_indexes.json'):
                processed_state_indexes = {col: i for i, col in enumerate(df.columns)}
                with open(processed_data_dir + f'processed_state_indexes.json', 'w') as f:
                    json.dump(processed_state_indexes, f)


def resampling(data: np.array, indexes: Dict, prediction_horizon: int) -> np.array:
    """
    :param data: normalized data set containing the setpoint_change feature
    :param indexes: state indexes matching columns names and position
    :param prediction_horizon: prediction horizon of the models trained with these data
    :return: resampled data set

    Resample the data around set point changes and return the resampled data set
    This function is made to only keep data collected during the transient phases (around temperature setpoint changes)
    """

    delta_set_point_indexes = find_delta_action_index(indexes)
    delta_set_point_indexes.sort()
    delta_set_point_index = delta_set_point_indexes[0]  # keep the first delta action, not the lags'

    set_point_changes = np.where(data[:, delta_set_point_index] != 0)[0]
    resampled_data = []
    last_end = 0

    for i in set_point_changes:
        start = max(0, i - prediction_horizon)
        end = min(data.shape[0], i + 2 * prediction_horizon)
        if start >= last_end:  # Add only non-overlapping ranges
            resampled_data.extend(data[start:end])
            last_end = end

    return np.array(resampled_data)


def add_features(
    data_file: pd.DataFrame,
    state_indexes: Dict,
    setpoint_change: bool,
    n_lags: int,
    simulation_time_step: float,
    include_action_lags: bool
) -> pd.DataFrame:
    """
    :param data_file: simulation data to process
    :param state_indexes: state indexes matching columns names and position
    :param setpoint_change: whether to include the difference between two consecutive temperature setpoints in the data
    :param n_lags: number of past observations to include to the lags
    :param simulation_time_step: EnergyPlus simulation timestep in minutes
    :param include_action_lags: whether to include lags of actions in the observations
    :return: data set containing the new features: lags of observations, calendar cyclical features, setpoint changes
    """

    df = pd.DataFrame(data_file, columns=state_indexes.keys())
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 6.0)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 6.0)
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
    if setpoint_change:
        df['setpoint_change_t-1'] = df['setpoint_change'].shift(1)
        df['delta_setpoint_change'] = df['setpoint_change'] - df['setpoint_change_t-1']
        df.drop('setpoint_change_t-1', axis=1, inplace=True)
    new_cols = []
    for lag in range(1, n_lags + 1):
        new_df = pd.DataFrame()
        new_df[f'timestamp_lag_{lag}'] = np.ones(df.shape[0]) * (n_lags - lag) * simulation_time_step
        new_df[f'hvac_power_lag_{lag}'] = df['hvac_power'].shift(lag)
        new_df[f'indoor_temperature_lag_{lag}'] = df['indoor_temperature'].shift(lag)
        if include_action_lags:
            new_df[f'outdoor_temperature_lag_{lag}'] = df['outdoor_temperature'].shift(lag)
            new_df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
            new_df[f'beam_solar_rad_lag_{lag}'] = df['beam_solar_rad'].shift(lag)
            new_df[f'day_of_week_sin_lag_{lag}'] = df['day_of_week_sin'].shift(lag)
            new_df[f'day_of_week_cos_lag_{lag}'] = df['day_of_week_cos'].shift(lag)
            new_df[f'hour_of_day_sin_lag_{lag}'] = df['hour_of_day_sin'].shift(lag)
            new_df[f'hour_of_day_cos_lag_{lag}'] = df['hour_of_day_cos'].shift(lag)
            new_df[f'setpoint_change_lag_{lag}'] = df['setpoint_change'].shift(lag)
            if setpoint_change:
                new_df[f'delta_setpoint_change_lag_{lag}'] = df['delta_setpoint_change'].shift(lag)
        new_cols.append(new_df)
    df = pd.concat([df] + new_cols, axis=1)
    df.drop(['day_of_week', 'hour_of_day'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df


def standardize_data(
    number_of_zones: int,
    processed_data_dir: str,
    data_indexes: list,
    state_indexes: dict,
    setpoint_change: bool,
    data_type: str,
    prediction_horizon: int,
    resample_data: bool
) -> None:
    """
    :param number_of_zones: number of zones in the building
    :param processed_data_dir: path to the processed data to standardize
    :param data_indexes: data indexes matching columns names and position
    :param state_indexes: state indexes matching columns names and position
    :param setpoint_change: whether to include the difference between two consecutive temperature setpoints in the data
    :param data_type: training, validation or test data
    :param prediction_horizon: prediction horizon of the models trained with these data
    :param resample_data: whether data have been resampled around temperature setpoint changes
    :return: None

    Normalize the data using (data - mean) / std for all the features except cyclical ones.
    Results are saved in numpy a csv file. The header is the index saved in a separate json file.
    """

    for zone_id in range(number_of_zones):
        processed_data = []
        for idx, i in enumerate(data_indexes):
            filename = f'zone{zone_id}/processed_data{idx}.csv'
            filepath = os.path.join(processed_data_dir, filename)
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            processed_data.append(data)
        processed_data = np.vstack(processed_data)
        if data_type == 'training':
            mean = np.mean(processed_data, axis=0)
            std = np.std(processed_data, axis=0)
            save_norm_constants(zone_id=zone_id, mean=mean, std=std, processed_data_dir=processed_data_dir)
        else:
            mean, std = load_norm_constants(zone_id=zone_id, processed_data_dir=processed_data_dir)
        # homogenize mean and std for power and temperature lags
        power_indexes = find_index_power(state_indexes)
        timestamp_indexes = find_index_timestamp(state_indexes)
        indoor_temperature_indexes = find_index_indoor_temperature(state_indexes)
        outdoor_temperature_indexes = find_index_outdoor_temperature(state_indexes)
        humidity_indexes = find_index_humidity(state_indexes)
        solar_rad_indexes = find_index_beam_solar_rad(state_indexes)

        cyclical_features_indexes = find_index_cyclical_features(state_indexes)
        delta_set_point_index = find_delta_action_index(state_indexes)
        mean[power_indexes] = np.ones(len(power_indexes)) * mean[state_indexes['hvac_power']]
        mean[indoor_temperature_indexes] = np.ones(len(indoor_temperature_indexes)) * mean[
            state_indexes['indoor_temperature']]
        mean[outdoor_temperature_indexes] = np.ones(len(outdoor_temperature_indexes)) * mean[
            state_indexes['outdoor_temperature']]
        mean[humidity_indexes] = np.ones(len(humidity_indexes)) * mean[state_indexes['humidity']]
        mean[solar_rad_indexes] = np.ones(len(solar_rad_indexes)) * mean[state_indexes['beam_solar_rad']]
        std[power_indexes] = np.ones(len(power_indexes)) * std[state_indexes['hvac_power']]
        std[indoor_temperature_indexes] = np.ones(len(indoor_temperature_indexes)) * std[
            state_indexes['indoor_temperature']]
        std[outdoor_temperature_indexes] = np.ones(len(outdoor_temperature_indexes)) * std[
            state_indexes['outdoor_temperature']]
        std[humidity_indexes] = np.ones(len(humidity_indexes)) * std[state_indexes['humidity']]
        std[solar_rad_indexes] = np.ones(len(solar_rad_indexes)) * std[state_indexes['beam_solar_rad']]
        # don't normalize cyclical features and set point change
        mean[cyclical_features_indexes] = np.zeros(len(cyclical_features_indexes))
        std[cyclical_features_indexes] = np.ones(len(cyclical_features_indexes))

        if setpoint_change:
            mean[delta_set_point_index] = 0
            std[delta_set_point_index] = 1

        mean[timestamp_indexes] = 0
        std[timestamp_indexes] = 1
        mean_matrix = np.tile(mean, (processed_data.shape[0], 1))
        std_matrix = np.tile(std, (processed_data.shape[0], 1))
        norm_data = (processed_data - mean_matrix) / std_matrix

        if resample_data:
            norm_data = resampling(norm_data, state_indexes, prediction_horizon)

        if not os.path.isdir(processed_data_dir + f'/zone{zone_id}'):
            os.makedirs(processed_data_dir + f'/zone{zone_id}')

        np.savetxt(processed_data_dir + f'/zone{zone_id}/{data_type}.csv', norm_data, delimiter=',')


def parse_base_simulation_results(
    datasets: List[pd.DataFrame],
    zone_id: int,
    heating_season: bool
) -> (List[np.array], Dict[str, int]):
    """
    :param datasets: dataset to parse. The data must have already been preprocessed.
    :param zone_id: id of the zone corresponding to the data to parse
    :param heating_season: whether data were collected during winter (heating season)
    :return: parsed simulation data (in numpy arrays and header indexes) for a given zone in a building
    """

    state_indexes = {
        'hvac_power': 0,
        'indoor_temperature': 1,
        'outdoor_temperature': 2,
        'humidity': 3,
        'beam_solar_rad': 4,
        'day_of_week': 5,
        'hour_of_day': 6,
        'setpoint_change': 7
    }

    preprocessed_data = []
    for data in datasets:
        hvac_power = data[f'hvac_power_zone{zone_id}'].to_numpy()
        temperature = data[f'temperature_zone{zone_id}'].to_numpy()
        # setpoint = data[f'temperature_setpoint_zone{zone_id}'].to_numpy()  # setpoints are often constant
        if heating_season:
            setpoint_change = data[f'heating_setpoint_change_zone{zone_id}'].to_numpy()
        else:
            setpoint_change = data[f'cooling_setpoint_change_zone{zone_id}'].to_numpy()
        # weather
        weather_features = [
            'outdoor_temperature',
            'humidity',
            'beam_solar_rad'
        ]
        weather = np.asarray(data[weather_features]).transpose()
        # calendar
        calendar_features = [
            'day_of_week',
            'hour_of_day'
        ]
        calendar = np.asarray(data[calendar_features]).transpose()
        # zone information
        # make sure the order of this line match the state indexes
        preprocessed_data.append(
            np.vstack([hvac_power, temperature, weather, calendar, setpoint_change]).transpose()
        )

    return preprocessed_data, state_indexes


def find_index_timestamp(state_indexes):
    return [value for key, value in state_indexes.items() if 'timestamp' in key]


def find_index_power(state_indexes):
    return [value for key, value in state_indexes.items() if 'power' in key]


def find_index_indoor_temperature(state_indexes):
    return [value for key, value in state_indexes.items() if 'indoor_temperature' in key]


def find_index_outdoor_temperature(state_indexes):
    return [value for key, value in state_indexes.items() if 'outdoor_temperature' in key]


def find_index_humidity(state_indexes):
    return [value for key, value in state_indexes.items() if 'humidity' in key]


def find_index_beam_solar_rad(state_indexes):
    return [value for key, value in state_indexes.items() if 'solar_rad' in key]


def find_index_cyclical_features(state_indexes):
    return [value for key, value in state_indexes.items() if 'sin' in key or 'cos' in key]


def find_delta_action_index(state_indexes):
    indexes = [value for key, value in state_indexes.items() if 'delta' in key]
    return indexes
