#!/bin/bash

for i in {1..7}
do
    python main.py base_simulation -sim_id $i -controller_id 1
done

python main.py train_test_models -m 'SSM' -sim_id 1 -b 'house' --zone_id 0 --base_config 1 --training --training_config 1 --test --test_config 1 --model_configs 1 --processing_data --processing_data_config 1

python main.py single_zone_building -sim_id 1 -m SSM --model_config 1 -c ShootingController --controller_config 1 --training_config 1
