import os
import glob
import gin
import time
import itertools


def count_config_files(directory):
    filepath = os.path.join(directory, 'config*.gin')
    config_files = glob.glob(filepath)
    return len(config_files)


def modify_and_parse_test_config(
    base_config_path: str,
    model_config_id: int,
    base_config_id: int,
    model_type: str
) -> None:
    """
    :param base_config_path: path to the training_test configuration file
    :param model_config_id: number of the configuration file to parse
    :param base_config_id: number of the base train test configuration to parse
    :param model_type: name of the model class to use

    Load the training and test configuration with the right model
    """
    with open(base_config_path, "r") as file:
        original_config = file.readlines()

    modified_config = f"include './state_space_models/configurations/train_test/base_config{base_config_id}.gin'\n"
    modified_config += f"include './state_space_models/configurations/models/{model_type}/config{model_config_id}.gin'\n"
    for line in original_config:
        if line.startswith("test.model ="):
            modified_config += f"test.model = @{model_type}\n"
        else:
            modified_config += line

    # Save the modified configuration to a temporary file, parse config does not handle include statements
    nanoseconds = time.time_ns()
    temp_filename = f"temp_config_{nanoseconds}.gin"  # create a unique file for parallel jobs
    with open(temp_filename, "w") as temp_file:
        temp_file.write(modified_config)

    gin.parse_config_file(temp_filename)
    os.remove(temp_filename)


def modify_and_parse_training_config(
    base_config_path: str,
    model_config_id: int,
    base_config_id: int,
    model_type: str
) -> None:
    """
    :param base_config_path: path to the training_test configuration file
    :param model_config_id: number of the configuration file to parse
    :param base_config_id: number of the base train test configuration to parse
    :param model_type: name of the model class to use

    Load the training and test configuration with the right model
    """
    with open(base_config_path, "r") as file:
        original_config = file.readlines()

    modified_config = f"include './state_space_models/configurations/train_test/base_config{base_config_id}.gin'\n"
    modified_config += f"include './state_space_models/configurations/models/{model_type}/config{model_config_id}.gin'\n"
    for line in original_config:
        if line.startswith("training.model ="):
            modified_config += f"training.model = @{model_type}\n"
        else:
            modified_config += line

    # Save the modified configuration to a temporary file, parse config does not handle include statements
    nanoseconds = time.time_ns()
    temp_filename = f"temp_config_{nanoseconds}.gin"  # create a unique file for parallel jobs
    with open(temp_filename, "w") as temp_file:
        temp_file.write(modified_config)

    gin.parse_config_file(temp_filename)
    os.remove(temp_filename)


def grid_search_rssm(base_config_path: str, start_number: int = None):
    """
    :param base_config_path:
    :param start_number:
    :return:
    """

    grid = {
        'seed': [1, 2, 3, 4, 5],
        'determinisitc_size': [100, 200, 400],
        'stochastic_size': [15, 30, 100],
        'learning_rate': [0.003, 0.0003, 0.00003],
        'loss_kl_scale': [0.1, 1.0, 10.0]
    }

    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    output_directory = os.path.dirname(base_config_path)
    start_number = count_config_files(output_directory) + 1 if start_number is None else start_number
    for i, experiment in enumerate(experiments, start=start_number):
        with open(base_config_path, "r") as file:
            original_config = file.readlines()
        modified_config = ""
        for line in original_config:
            if line.startswith("RSSM.latent_state_size ="):
                modified_config += f"RSSM.latent_state_size = {experiment['determinisitc_size']}\n"
            elif line.startswith("RSSM.determinisitc_size ="):
                modified_config += f"RSSM.determinisitc_size = {experiment['determinisitc_size']}\n"
            elif line.startswith("RSSM.stochastic_size ="):
                modified_config += f"RSSM.stochastic_size = {experiment['stochastic_size']}\n"
            elif line.startswith("RSSM.learning_rate ="):
                modified_config += f"RSSM.learning_rate = {experiment['learning_rate']}\n"
            elif line.startswith("RSSM.loss_kl_scale ="):
                modified_config += f"RSSM.loss_kl_scale = {experiment['loss_kl_scale']}\n"
            elif line.startswith("RSSM.seed ="):
                modified_config += f"RSSM.seed = {experiment['seed']}\n"
            else:
                modified_config += line
        new_config_filename = os.path.join(output_directory, f"config{i}.gin")
        with open(new_config_filename, "w") as new_config_file:
            new_config_file.write(modified_config)


def grid_search_ssm(base_config_path: str, start_number: int = None):
    """
    :param base_config_path:
    :param start_number:
    :return:
    """

    grid = {
        'seed': [1, 2, 3, 4, 5],
        'latent_state_size': [50, 100, 250, 500],
        'learning_rate': [0.003, 0.0003, 0.00003]
    }
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    output_directory = os.path.dirname(base_config_path)
    start_number = count_config_files(output_directory) + 1 if start_number is None else start_number
    for i, experiment in enumerate(experiments, start=start_number):
        with open(base_config_path, "r") as file:
            original_config = file.readlines()
        modified_config = ""
        for line in original_config:
            if line.startswith("SSM.latent_state_size ="):
                modified_config += f"SSM.latent_state_size = {experiment['latent_state_size']}\n"
            elif line.startswith("SSM.encoder_dim ="):
                modified_config += f"SSM.encoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("SSM.decoder_dim ="):
                modified_config += f"SSM.decoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("SSM.learning_rate ="):
                modified_config += f"SSM.learning_rate = {experiment['learning_rate']}\n"
            elif line.startswith("SSM.seed ="):
                modified_config += f"SSM.seed = {experiment['seed']}\n"
            else:
                modified_config += line
        new_config_filename = os.path.join(output_directory, f"config{i}.gin")
        with open(new_config_filename, "w") as new_config_file:
            new_config_file.write(modified_config)


def grid_search_structured_ssm(base_config_path: str, start_number: int = None):
    """
    :param base_config_path:
    :return:
    """

    grid = {
        'seed': [1, 2, 3, 4, 5],
        'latent_size': [50, 100, 150, 200, 250, 300],
        'learning_rate': [0.003, 0.0003]
    }
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    output_directory = os.path.dirname(base_config_path)
    start_number = count_config_files(output_directory) + 1 if start_number is None else start_number
    for i, experiment in enumerate(experiments, start=start_number):
        with open(base_config_path, "r") as file:
            original_config = file.readlines()
        modified_config = ""
        for line in original_config:
            if line.startswith("StructuredSSM.latent_size ="):
                modified_config += f"StructuredSSM.latent_size = {experiment['latent_size']}\n"
            elif line.startswith("StructuredSSM.encoder_dim ="):
                modified_config += f"StructuredSSM.encoder_dim = {experiment['latent_size']}\n"
            elif line.startswith("StructuredSSM.decoder_dim ="):
                modified_config += f"StructuredSSM.decoder_dim = {experiment['latent_size']}\n"
            elif line.startswith("StructuredSSM.learning_rate ="):
                modified_config += f"StructuredSSM.learning_rate = {experiment['learning_rate']}\n"
            elif line.startswith("StructuredSSM.seed ="):
                modified_config += f"StructuredSSM.seed = {experiment['seed']}\n"
            else:
                modified_config += line
        new_config_filename = os.path.join(output_directory, f"config{i}.gin")
        with open(new_config_filename, "w") as new_config_file:
            new_config_file.write(modified_config)


def grid_search_neural_ode(base_config_path: str, start_number: int = None):
    """
    :param base_config_path:
    :return:
    """

    grid = {
        'seed': [1, 2, 3, 4, 5],
        'latent_state_size': [50, 100, 200],
        'depth': [1, 2],
        'learning_rate': [0.003, 0.0003]
    }
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    output_directory = os.path.dirname(base_config_path)
    start_number = count_config_files(output_directory) + 1 if start_number is None else start_number
    for i, experiment in enumerate(experiments, start=start_number):
        with open(base_config_path, "r") as file:
            original_config = file.readlines()
        modified_config = ""
        for line in original_config:
            if line.startswith("NeuralODE.latent_state_size ="):
                modified_config += f"NeuralODE.latent_state_size = {experiment['latent_state_size']}\n"
            elif line.startswith("NeuralODE.depth ="):
                modified_config += f"NeuralODE.depth = {experiment['depth']}\n"
            elif line.startswith("NeuralODE.seed ="):
                modified_config += f"NeuralODE.seed = {experiment['seed']}\n"
            elif line.startswith("NeuralODE.learning_rate ="):
                modified_config += f"NeuralODE.learning_rate = {experiment['learning_rate']}\n"
            else:
                modified_config += line
        new_config_filename = os.path.join(output_directory, f"config{i}.gin")
        with open(new_config_filename, "w") as new_config_file:
            new_config_file.write(modified_config)


def grid_search_latent_neural_ode(base_config_path: str, start_number: int = None):
    """
    :param base_config_path:
    :return:
    """

    grid = {
        'seed': [1, 2, 3, 4, 5],
        'latent_state_size': [50, 100, 200],
        'vf_depth': [1, 2],
        'learning_rate': [0.003, 0.0003],
    }
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    output_directory = os.path.dirname(base_config_path)
    start_number = count_config_files(output_directory) + 1 if start_number is None else start_number
    for i, experiment in enumerate(experiments, start=start_number):
        with open(base_config_path, "r") as file:
            original_config = file.readlines()
        modified_config = ""
        for line in original_config:
            if line.startswith("LatentNeuralODE.latent_state_size ="):
                modified_config += f"LatentNeuralODE.latent_state_size = {experiment['latent_state_size']}\n"
            elif line.startswith("LatentNeuralODE.decoder_dim ="):
                modified_config += f"LatentNeuralODE.decoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("LatentNeuralODE.encoder_dim ="):
                modified_config += f"LatentNeuralODE.encoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("LatentNeuralODE.vf_depth ="):
                modified_config += f"LatentNeuralODE.vf_depth = {experiment['vf_depth']}\n"
            elif line.startswith("LatentNeuralODE.learning_rate ="):
                modified_config += f"LatentNeuralODE.learning_rate = {experiment['learning_rate']}\n"
            elif line.startswith("LatentNeuralODE.seed ="):
                modified_config += f"LatentNeuralODE.seed = {experiment['seed']}\n"
            else:
                modified_config += line
        new_config_filename = os.path.join(output_directory, f"config{i}.gin")
        with open(new_config_filename, "w") as new_config_file:
            new_config_file.write(modified_config)


def grid_search_structured_latent_neural_ode(base_config_path: str, start_number: int = None):
    """
    :param base_config_path:
    :return:
    """

    grid = {
        'seed': [1, 2, 3, 4, 5],
        'latent_state_size': [50, 100, 200],
        'vf_depth': [1, 2],
        'learning_rate': [0.003, 0.0003],
    }
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    output_directory = os.path.dirname(base_config_path)
    start_number = count_config_files(output_directory) + 1 if start_number is None else start_number
    for i, experiment in enumerate(experiments, start=start_number):
        with open(base_config_path, "r") as file:
            original_config = file.readlines()
        modified_config = ""
        for line in original_config:
            if line.startswith("StructuredLatentNeuralODE.latent_state_size ="):
                modified_config += f"StructuredLatentNeuralODE.latent_state_size = {experiment['latent_state_size']}\n"
            elif line.startswith("StructuredLatentNeuralODE.decoder_dim ="):
                modified_config += f"StructuredLatentNeuralODE.decoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("StructuredLatentNeuralODE.encoder_dim ="):
                modified_config += f"StructuredLatentNeuralODE.encoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("StructuredLatentNeuralODE.vf_depth ="):
                modified_config += f"StructuredLatentNeuralODE.vf_depth = {experiment['vf_depth']}\n"
            elif line.startswith("StructuredLatentNeuralODE.learning_rate ="):
                modified_config += f"StructuredLatentNeuralODE.learning_rate = {experiment['learning_rate']}\n"
            elif line.startswith("StructuredLatentNeuralODE.seed ="):
                modified_config += f"StructuredLatentNeuralODE.seed = {experiment['seed']}\n"
            else:
                modified_config += line
        new_config_filename = os.path.join(output_directory, f"config{i}.gin")
        with open(new_config_filename, "w") as new_config_file:
            new_config_file.write(modified_config)


def grid_search_ce_ssm(base_config_path: str, start_number: int = None):
    """
    :param base_config_path:
    :return:
    """

    grid = {
        'seed': [1, 2, 3, 4, 5],
        'latent_state_size': [50, 100, 200, 300],
        'learning_rate': [0.003, 0.003, 0.03],
    }
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_directory = os.path.dirname(base_config_path)
    start_number = count_config_files(output_directory) + 1 if start_number is None else start_number
    for i, experiment in enumerate(experiments, start=start_number):
        with open(base_config_path, "r") as file:
            original_config = file.readlines()
        modified_config = ""
        for line in original_config:
            if line.startswith("CESSM.latent_state_size ="):
                modified_config += f"CESSM.latent_state_size = {experiment['latent_state_size']}\n"
            elif line.startswith("CESSM.decoder_dim ="):
                modified_config += f"CESSM.decoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("CESSM.encoder_dim ="):
                modified_config += f"CESSM.encoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("CESSM.learning_rate ="):
                modified_config += f"CESSM.learning_rate = {experiment['learning_rate']}\n"
            elif line.startswith("CESSM.seed ="):
                modified_config += f"CESSM.seed = {experiment['seed']}\n"
            else:
                modified_config += line
        new_config_filename = os.path.join(output_directory, f"config{i}.gin")
        with open(new_config_filename, "w") as new_config_file:
            new_config_file.write(modified_config)


def grid_search_ce_latent_neural_ode(base_config_path: str, start_number: int = None):
    """
    :param base_config_path:
    :return:
    """

    grid = {
        'seed': [1, 2, 3, 4, 5],
        'latent_state_size': [50, 100, 200],
        'vf_depth': [1, 2],
        'learning_rate': [0.003, 0.0003],
    }
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_directory = os.path.dirname(base_config_path)
    start_number = count_config_files(output_directory) + 1 if start_number is None else start_number
    for i, experiment in enumerate(experiments, start=start_number):
        with open(base_config_path, "r") as file:
            original_config = file.readlines()
        modified_config = ""
        for line in original_config:
            if line.startswith("CELatentNeuralODE.latent_state_size ="):
                modified_config += f"CELatentNeuralODE.latent_state_size = {experiment['latent_state_size']}\n"
            elif line.startswith("CELatentNeuralODE.decoder_dim ="):
                modified_config += f"CELatentNeuralODE.decoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("CELatentNeuralODE.encoder_dim ="):
                modified_config += f"CELatentNeuralODE.encoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("CELatentNeuralODE.vf_depth ="):
                modified_config += f"CELatentNeuralODE.vf_depth = {experiment['vf_depth']}\n"
            elif line.startswith("CELatentNeuralODE.learning_rate ="):
                modified_config += f"CELatentNeuralODE.learning_rate = {experiment['learning_rate']}\n"
            elif line.startswith("CELatentNeuralODE.seed ="):
                modified_config += f"CELatentNeuralODE.seed = {experiment['seed']}\n"
            else:
                modified_config += line
        new_config_filename = os.path.join(output_directory, f"config{i}.gin")
        with open(new_config_filename, "w") as new_config_file:
            new_config_file.write(modified_config)


def grid_search_cde(base_config_path: str, start_number: int = None):

    grid = {
        'seed': [1, 2, 3, 4, 5],
        'latent_state_size': [50, 100, 200],
        'vf_depth': [1, 2],
        'learning_rate': [0.003, 0.0003]
    }
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_directory = os.path.dirname(base_config_path)
    start_number = count_config_files(output_directory) + 1 if start_number is None else start_number
    for i, experiment in enumerate(experiments, start=start_number):
        with open(base_config_path, "r") as file:
            original_config = file.readlines()
        modified_config = ""
        for line in original_config:
            if line.startswith("CDE.latent_state_size ="):
                modified_config += f"CDE.latent_state_size = {experiment['latent_state_size']}\n"
            elif line.startswith("CDE.decoder_dim ="):
                modified_config += f"CDE.decoder_dim = {experiment['latent_state_size']}\n"
            elif line.startswith("CDE.init_encoder_width_size ="):
                modified_config += f"CDE.init_encoder_width_size = {experiment['latent_state_size']}\n"
            elif line.startswith("CDE.model_width_size ="):
                modified_config += f"CDE.model_width_size = {experiment['latent_state_size']}\n"
            elif line.startswith("CDE.model_depth ="):
                modified_config += f"CDE.model_depth = {experiment['vf_depth']}\n"
            elif line.startswith("CDE.learning_rate ="):
                modified_config += f"CDE.learning_rate = {experiment['learning_rate']}\n"
            elif line.startswith("CDE.seed ="):
                modified_config += f"CDE.seed = {experiment['seed']}\n"
            else:
                modified_config += line
        new_config_filename = os.path.join(output_directory, f"config{i}.gin")
        with open(new_config_filename, "w") as new_config_file:
            new_config_file.write(modified_config)


def create_configs(model_type: str, start_number: int = None) -> None:

    if model_type == 'SSM':
        grid_search_ssm(
            f'./state_space_models/configurations/models/{model_type}/base_config.gin',
            start_number=start_number
        )
    elif model_type == 'RSSM':
        grid_search_rssm(
            f'./state_space_models/configurations/models/{model_type}/base_config.gin',
            start_number=start_number
        )
    elif model_type == 'StructuredSSM':
        grid_search_structured_ssm(
            f'./state_space_models/configurations/models/{model_type}/base_config.gin',
            start_number=start_number
        )
    elif model_type == 'LatentNeuralODE':
        grid_search_latent_neural_ode(
            f'./state_space_models/configurations/models/{model_type}/base_config.gin',
            start_number=start_number
        )
    elif model_type == 'StructuredLatentNeuralODE':
        grid_search_structured_latent_neural_ode(
            f'./state_space_models/configurations/models/{model_type}/base_config.gin',
            start_number=start_number
        )
    elif model_type == 'NeuralODE':
        grid_search_neural_ode(
            f'./state_space_models/configurations/models/{model_type}/base_config.gin',
            start_number=start_number
        )
    elif model_type == 'CESSM':
        grid_search_ce_ssm(
            f'./state_space_models/configurations/models/{model_type}/base_config.gin',
            start_number=start_number
        )
    elif model_type == "CELatentNeuralODE":
        grid_search_ce_latent_neural_ode(
            f"./state_space_models/configurations/models/{model_type}/base_config.gin",
            start_number=start_number
        )
    elif model_type == 'CDE':
        grid_search_cde(
            f"./state_space_models/configurations/models/{model_type}/base_config.gin",
            start_number=start_number
        )
    else:
        raise ValueError(f'model type {model_type} not handle for grid search')
