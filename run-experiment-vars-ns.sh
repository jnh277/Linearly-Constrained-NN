#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=10:00:00
#PBS -l software=torch
#PBS -k oe

source /etc/profile.d/modules.sh
module load torch/1.0.1-python3.6

cd /home/jnh277/Linearly-Constrained-NN

echo "Running with Z=$Z and V=$V and I=$I"

python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 500 --n_train 6000 --save_file mag_data_netsize2/exp_${Z}_trial_${I} --net_hidden_size ${Z} ${V}


#python experiment_2D.py --n_data 4000 --net_hidden_size ${Z} ${V} --save_file net_size_study200/exp_${Z}_trial_${I} --epochs 400 --scheduler 1