#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=10:00:00           
#PBS -l software=torch
#PBS -k oe

source /etc/profile.d/modules.sh
module load torch/1.0.1-python3.6  
 
cd /home/jnh277/Linearly-Constrained-NN

echo "Running with ND=$ND and I=$I"

python experiment_2D.py --n_data ${ND} --save_file n_data_study/exp_${ND}_trial_${I} --epochs 400 --scheduler 1 --display

