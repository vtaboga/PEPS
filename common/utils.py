from typing import List, Dict
import pandas as pd
import numpy as np
import json
import os
import gin
import tempfile


def save_config_file(config_str, results_dir, base_filename='configuration'):
    # Start with the base filename
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    filename = results_dir + '/' + base_filename + '.gin'
    counter = 1
    while os.path.exists(filename):
        filename = results_dir + '/' + f'{base_filename}{counter}.gin'
        counter += 1
    with open(filename, 'w') as f:
        f.write(config_str)


def get_log_filename(results_dir: str, base_filename='project_gradient_logs') -> str:
    """
    Determines a unique filename in the given directory based on the base_filename.
    If a file with the base name doesn't exist, it returns the base name.
    Otherwise, it appends a counter to the base name.

    Args:
        results_dir (str): Directory where the log file is to be saved.
        base_filename (str, optional): Base filename for the log file. Defaults to 'project_gradient_logs'.

    Returns:
        str: Full path of the determined unique filename.
    """

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    extension = ".txt"
    filename = os.path.join(results_dir, base_filename + extension)

    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(results_dir, f"{base_filename}{counter}{extension}")
        counter += 1

    return filename


def save_control_memory(data, results_path, base_filename='control_memory', final: bool = True):
    """
    Save the DataFrame to a unique CSV filename.

    Parameters:
    - data (pd.DataFrame): The DataFrame to save.
    - results_path (str): The path to the directory where the file should be saved.
    - base_filename (str): The base name of the file (without the .csv extension).
    """

    temp_path = os.path.join(results_path, 'temp_control_memory.csv')

    if final:
        counter = 0
        filename = base_filename + '.csv'
        full_path = os.path.join(results_path, filename)

        while os.path.exists(full_path):
            counter += 1
            filename = f"{base_filename}{counter}.csv"
            full_path = os.path.join(results_path, filename)

        data.to_csv(full_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

    else:
        data.to_csv(temp_path)


def save_results(
    results: Dict,
    zone_id: int,
    results_dir: str,
    seed: int = None,
    validation: bool = False
) -> None:

    data_type = 'validation' if validation else 'test'
    if not os.path.isdir(results_dir + f'/zone{zone_id}'):
        os.makedirs(results_dir + f'/zone{zone_id}')
    # covert results to numpy arrays
    results = {k: np.array(v) for k, v in results.items()}
    results = pd.DataFrame(results)
    if seed is None:
        results.to_csv(results_dir + f'/{data_type}_results.csv')
    else:
        results.to_csv(results_dir + f'/{data_type}_results_seed_{seed}.csv')


def get_normalization_constants(
    zone_ids: List,
    simulation_number: int,
    config_number: int,
    building: str
) -> (List[List[float]], List[List[float]], Dict):
    """zone_ids : list of zone ids for which to fetch the constants
    return a dict containing the normalization constants"""

    path = f'./off_line_computation/data/{building}/simulation_{simulation_number}/config_{config_number}'
    means = []
    stds = []
    for zone_id in zone_ids:
        f = open(path + f'/zone{zone_id}/mean.json')
        mean = json.load(f)
        means.append(mean)
        f.close()
        f = open(path + f'/zone{zone_id}/std.json')
        std = json.load(f)
        stds.append(std)
        f.close()
    f = open(path + f'/processed_state_indexes.json')
    state_indexes = json.load(f)

    return means, stds, state_indexes


def modify_idf_file(
    file_path: str,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
    simulation_timestep: int
):
    """
    Modify the RunPeriod and simulation timestep in an idf file dynamically, searching for specific comment tokens.

    :param file_path: Path to the template idf file.
    :param start_month: Month to start the simulation.
    :param start_day: Day of the month to start the simulation.
    :param end_month: Month to end the simulation.
    :param end_day: Day of the month to end the simulation.
    :param simulation_timestep: Simulation time step.
    :return: Path to the temporary modified idf file.
    """

    # Create a temporary file
    temp_file_dir = tempfile.gettempdir()
    temp_file_name = next(tempfile._get_candidate_names()) + ".idf"
    temp_file_path = os.path.join(temp_file_dir, temp_file_name)

    try:
        with open(file_path, 'r') as file, open(temp_file_path, 'w') as temp_file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if '! ------ RunPeriod ------' in line:  # RunPeriod token
                    lines[i+3] = f"    {start_month},                       !- Begin Month\n"
                    lines[i+4] = f"    {start_day},                      !- Begin Day of Month\n"
                    lines[i+6] = f"    {end_month},                       !- End Month\n"
                    lines[i+7] = f"    {end_day},                      !- End Day of Month\n"
                if '! ------ Timestep ------' in line:  # Timestep token
                    lines[i+1] = f"Timestep,{simulation_timestep};\n"
            temp_file.writelines(lines)

    except IOError as e:
        print(f"Error: {e}")
        return None

    return temp_file_path
