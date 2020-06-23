#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=30:00:00
#PBS -l software=torch
#PBS -k oe

source /etc/profile.d/modules.sh
#module load torch/1.1.0-python3.6
module load torch/1.0.1-python3.6

cd /home/jnh277/Linearly-Constrained-NN

echo "Running with w=$W and I=$I"

python pointObsComparison.py  --constraint_weighting ${W} --save_file pointObsStudy/exp_${W}_trial_${I} --epochs 400 --scheduler 1 --display
