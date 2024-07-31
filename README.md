# PEPS
Python-EnergyPlus Simulator

This repository contains control algorithms for HVAC management in buildings. The simulations are made with EnergyPlus and the control is handled in Python using the EnergyPlus - Python API.

## Setting things up

- Install the requirements: `python -m pip install -r requirements.txt` (Python 3.10)
- Install EnergyPlus (Version 22-1-0): [EnergyPlus Downloads](https://energyplus.net/downloads) (make sure to note the path where EnergyPlus is installed!)
- Add the EnergyPlus path to the configuration in `common/configuration/path_for_energyplus.json`

### Computation on GPU
If you wish to use a GPU (useful for training the deep learning models, not that much for running the simulation), do not use the requirements file and consult the JAX documentation for GPU support. Make sure to install Haiku using `pip install git+https://github.com/deepmind/dm-haiku` to avoid installing JAX for CPUs by default.

## Quick start

To run a simulation with random setpoint changes to collect data:

```bash
python main.py base_simulation -sim_id 1 -controller_id 1
```

To train and test the zones' models

```bash 
python main.py train_test_models -m 'SSM' -sim_id 1 -b 'house' --zone_id 0 --base_config 1 --training --training_config 1 --test --test_config 1 --model_configs 1 2 --processing_data --processing_data_config 1
```

You may also create configuration files using the arguments (be careful, it may override some configurations files) :

```bash
--create_model_configs --start_config_number 1
```

Once models have been trained, you may run a simulation with one of the control algorithms. For the single zone case:
 
```bash
python main.py single_zone_building -sim_id 1 -m SSM --model_config 1 -c ShootingController --controller_config 1 --training_config 1
```

## Organization of the repository

- **energyplus_simulator:** Contains everything related to EnergyPlus. For every building corresponding to a different .idf file (stored in the `buildings_blueprint` folder), a class inheriting from the `EnergyPlusAbstractBuilding` class must be created. This class is used to specify the name of the sensors and actuators as well as what is included in the power measurements. Weather files in .tmy format to run the simulations should be added to the `weather` directory.

- **state_space_models:** Contains the definition of the deep learning models to forecast a zone's temperature and HVAC power consumption. The models' structures, such as the neural network architectures, are defined in the `base_models`. Each model should be a class inheriting from `ABCModel`. This abstract class defines the training and test methods as well as the method to run the model to forecast the state evolution. All the data should be normalized in the models. The `Normalizer` class defined in the `utils` is used for this purpose. The mean and std values of the training data should be given as input to the models. Note that the recurrent state space model has a different training procedure, and the training method is thus overwritten in `RSSM` directly. The models' configurations are specified by a .gin file stored in the `configuration` folder.

- **controllers:** Contains the definition of the zones' controllers. The simplest one is the `RandomController` which randomly changes the heating or cooling setpoint (depending on the season and the building's system). The other types of models, i.e., `ShootingController` and `RobustController`, use one of the deep learning models for planning. Deterministic models should be used with the shooting method while the recurrent state space model should be used with the stochastic robust control. All controllers should inherit from `ABCController` and the configurations should be specified in the `configurations` folder. Classes `RayShootingController` and `RayRobustController` embed the controller classes for usage with Ray for parallel computing. Note that configurations for Ray controllers are defined in .json files and not in .gin files to avoid parsing issues between Ray actors and gin.

 - **building_coordinator** Contains the definition of the zones' coordinator. A coordinator should be used to supervise local controllers acting in each zone of a multizone building. The distributed coordinator is based on ADMM and distributes a total heating/cooling power budget to each local controller. To benefit from the distributed optimization, computation should be done in parallel using Ray. To this aim, the `RayShootingController` and `RayRobustController` local controllers may be used.

- **common:** Binds everything together to run a simulation with the control. System classes define the callbacks for the EnergyPlus simulation. Values may be read from sensors and actions passed from the controllers to the EnergyPlus actuators in the system class. System classes are child classes of the `EnergyPlusWrapper` defined in `energyplus_simulator`. The vanilla `System` defines the abstract methods that every system must contain. For each type of control, a system class should be created. For instance, the `BaseSimulation` implements the random controls and is mostly used to collect data to train the deep learning models. `SingleZoneBuildingSimulation` implements a shooting control for a single zone building with a constraint on the maximum power usage. The configurations for the simulation are defined in .gin files in the `configurations` folder. Note that the power constraints are specified in `configurations/consumption_schedule`. You may look at the format of the existing schedules to create new ones.

- **main_scripts:** Contains the scripts to launch the simulations defined by the System classes. It contains the functions parsing the different .gin files according to each script's arguments. Note that each main script is called from `main.py`.

- **off_line_computation:** Contains the functions to process the data for training the deep learning models, as well as the scripts to run the training and tests of the models.

## Creating new models and simulation

- **Configurations:** To change the configurations, you may follow the examples given by the `base_config.gin` and create files named `config{i}.gin`, where i is a unique identifier of the configuration. The scripts in `/state_space_models/configurations/utils` may also be edited to create multiple configs for grid searches.

- **New buildings:** To add a new building, follow the examples given by `House` and `LowriseApartments`. Make sure to implement every abstract method. To identify the sensors and actuators name, refer to the EnergyPlus output files (.mdd mainly) and the .idf building definition.

## Singularity containers

- To create a singularity container, you may use the .def file `peps_container.def`. In a terminal, use the following command to create the container:

```bash
sudo singularity build peps_container.sif peps_container.def
```

The data and configurations are not directly included in the container, you should bind the container's directories to the directories containing the necessary configurations and data, as well as the directory you want to save the results to. For instance:

```bash
singularity exec -B ./models_config/SSM:/home/PEPS/state_space_models/configurations/models/SSM -B ./control_1:/home/PEPS/common/results/lowrise_apartments/simulation_1/control_results -B ./base_sim_results:/home/PEPS/common/results/lowrise_apartments/simulation_3 -B ./train_test_config:/PEPS/thermulator/state_space_models/configurations/train_test -B ./coordinator_config:/home/PEPS/building_coordinator/configurations/DistributedCoordinator -B ./simulation_config:/home/PEPS/common/configurations -B ./controllers_config:/home/PEPS/controllers/configurations  -B ./building/buildings_blueprint:/home/PEPS/energyplus_simulator/buildings_blueprint -B ./building/weather:/home/PEPS/energyplus_simulator/weather ./thermulator_cont.sif /bin/bash -c"
```

Containers are convenient for computation on a remote cluster and for reproducibility of the experiments. Refer to the Singularity documentation for more information.

## Handling time steps in EnergyPlus

```  --| beginning: observe set points and weather|---| end: observe energy consumption and temperature|--> ```

The weather and setpoint are observed at the beginning of each time step. 
The will causes changes in the zone's temperature and energy consumption, that are observed at the end of the time step.

Predictions must thus been made at the beginning of a time step. Let t- and t+ be the beginning and the end of time step t.
If pred is the prediction model, taking as input a set point sp, a weather w, a power P and a temperature T, we have:

```P(t1+), T(t1+) = pred(w(t1-), sp(t1-), P(t0+), W(t0+))```

The set points can be modified at any time during the time step, but mind the moment when update memory is called.
The current set point will be saved at this moment.

Note that this distinction is necessary only because the weather values are updated at the beginning of each time step,
and the power and the temperature are averaged over a time step.

## Associated papers

[Neural Differential Equations for Temperature Control in Buildings under Demand Response Programs](https://www.sciencedirect.com/science/article/pii/S030626192400816X), V. Taboga, C. Gehring, M. Le Cam, H. Dagdougui, P. L. Bacon.

[A Distributed ADMM-based Deep Learning Approach for Thermal Control in Multi-Zone Buildings under Demand Response Events](https://arxiv.org/abs/2312.05073), V. Taboga, H. Dagdougui.


