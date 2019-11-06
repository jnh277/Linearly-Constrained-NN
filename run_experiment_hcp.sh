#!/bin/bash

#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=10:00:00
#PBS -l software=torch
#PBS -k oe



source /etc/profile.d/modules.sh
module load torch/1.0.1-python3.6


cd /home/jnh277/Linearly-Constrained-NN

nd=4000
i=1


python experiment_2D.py --n_data ${nd} --save_file hcp/exp_${nd}_trial_${i} --epochs 2000 --scheduler 1