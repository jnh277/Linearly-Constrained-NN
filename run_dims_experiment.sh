#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=100:00:00
#PBS -l software=torch
#PBS -k oe

source /etc/profile.d/modules.sh
module load torch/1.0.1-python3.6

cd /home/jnh277/Linearly-Constrained-NN

echo "Running with D=$D and I=$I"

python experiment_high_D.py --epochs 600 --dims ${D} --save_file dims_study/exp_${D}_trial_${I} --n_data 50000 --display
