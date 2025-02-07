import argparse
from main_scripts.base_simulation import main as base_simulation
from main_scripts.train_test_models import main as train_test_models
from main_scripts.single_zone_building_control import main as single_zone_building_simulation


def run_base_simulation(args) -> None:
    """
    :param args:
    :return: None

    Call the base_simulation function
    """
    base_simulation(args.base_simulation_configuration_id, args.controller_configuration_id, args.controller_type)


def run_train_test_models(args) -> None:
    """
    :param args:
    :return: None

    Call the train_test_models function
    """

    for zone_id in args.zone_id:
        print(f'******* zone {zone_id} *******')
        train_test_models(
            training=args.training,
            test=args.test,
            create_model_configs=args.create_model_configs,
            start_config_number=args.start_config_number,
            simulation_id=args.base_simulation_configuration_id,
            processing_data=args.processing_data,
            processing_data_config=args.processing_data_config,
            model_type=args.model_type,
            building=args.building,
            zone_id=zone_id,
            training_config=args.training_config,
            test_config=args.test_config,
            base_config=args.base_config,
            model_configs=args.model_configs
        )


def run_single_zone_building_simulation(args) -> None:
    """
    :param args:
    :return:
    """
    single_zone_building_simulation(
        simulation_configuration=args.simulation_configuration,
        controller_type=args.controller_type,
        controller_config=args.controller_config,
        model_type=args.model_type,
        model_config=args.model_config,
        base_model_config=args.base_model_config,
        training_config=args.training_config
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='script', required=True)

    # Arguments for base_simulation
    parser_base_simulation = subparsers.add_parser('base_simulation')
    parser_base_simulation.add_argument('--base_simulation_configuration_id', '-sim_id', type=int, required=True)
    parser_base_simulation.add_argument('--controller_configuration_id', '-controller_id', type=int, required=True)
    parser_base_simulation.add_argument('--controller_type', type=str, default='RandomController')

    # Arguments for single_zone_building_control
    parser_single_zone_building = subparsers.add_parser('single_zone_building')
    parser_single_zone_building.add_argument('--simulation_configuration', '-sim_id', type=int, required=True)
    parser_single_zone_building.add_argument(
        "--model_type",
        '-m',
        type=str,
        required=False,
        choices=[
            'SSM',
            'RSSM',
            'StructuredSSM',
            'CESSM',
            'CDE',
            'LatentNeuralODE',
            'StructuredLatentNeuralODE',
            'CELatentNeuralODE',
            'NeuralODE',
        ],
    )
    parser_single_zone_building.add_argument("--model_config", "-m_config", type=int)
    parser_single_zone_building.add_argument("--training_config", "-t_config", type=int)
    parser_single_zone_building.add_argument("--base_model_config", type=int, default=1)
    parser_single_zone_building.add_argument(
        "--controller_type",
        "-c",
        type=str,
        choices=['ShootingController', 'RuleBasedController']
    )
    parser_single_zone_building.add_argument("--controller_config", "-c_config", type=int)

    # Arguments for train_test_models
    parser_train_test_models = subparsers.add_parser('train_test_models')
    parser_train_test_models.add_argument(
        "--model_type",
        '-m',
        type=str,
        required=False,
        choices=[
            'SSM',
            'RSSM',
            'StructuredSSM',
            'CESSM',
            'CDE',
            'LatentNeuralODE',
            'StructuredLatentNeuralODE',
            'CELatentNeuralODE',
            'NeuralODE',
        ],
    )
    parser_train_test_models.add_argument('--base_simulation_configuration_id', '-sim_id', type=int, required=True)
    parser_train_test_models.add_argument(
        "--building",
        '-b',
        type=str,
        required=True,
        help="Name of the building. This parameter should match the name of the folder in /data."
    )
    parser_train_test_models.add_argument("--zone_id", type=int, nargs='+', required=True)
    parser_train_test_models.add_argument("--base_config", type=int, default=1)
    parser_train_test_models.add_argument(
        "--model_configs",
        nargs='*',
        type=int,
        default=[1],
        help='model configs to train or test on'
    )
    parser_train_test_models.add_argument("--create_model_configs", action='store_true')
    parser_train_test_models.add_argument("--start_config_number", type=int, required=False, default=1)
    parser_train_test_models.add_argument("--training", action='store_true')
    parser_train_test_models.add_argument("--training_config", type=int, required=False)
    parser_train_test_models.add_argument("--test", action='store_true')
    parser_train_test_models.add_argument("--test_config", type=int, required=False)
    parser_train_test_models.add_argument("--processing_data", action='store_true')
    parser_train_test_models.add_argument("--processing_data_config", type=int, required=False, default=1)

    args = parser.parse_args()

    if args.script == 'base_simulation':
        run_base_simulation(args)
    elif args.script == 'train_test_models':
        # Check parse conditions
        if args.processing_data and args.processing_data_config is None:
            parser.error("--processing_data_config is required when --processing_data is specified.")
        if args.create_model_configs and args.start_config_number is None:
            parser.error("--start_config_number is required when --create_model_configs is specified.")
        if args.training and args.training_config is None:
            parser.error("--training_config is required when --training is specified.")
        if args.test and args.test_config is None:
            parser.error("--test_confing is required when --test is specified.")
        if (args.training or args.test) and args.model_type is None:
            parser.error("--model_type is required when --training is specified.")
        run_train_test_models(args)
    elif args.script == 'single_zone_building':
        run_single_zone_building_simulation(args)
    else:
        raise ValueError("Invalid script name")


if __name__ == '__main__':
    main()
