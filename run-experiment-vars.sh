#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=30:00:00
#PBS -l software=torch
#PBS -k oe

source /etc/profile.d/modules.sh
#module load torch/1.1.0-python3.6
module load torch/1.0.1-python3.6
 
cd /home/jnh277/Linearly-Constrained-NN

echo "Running with ND=$ND and I=$I"

#python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 500 --n_train ${ND} --save_file mag_data_n_study2/exp_${ND}_trial_${I} --net_hidden_size 150 75
#python experiment_2D.py --n_data ${ND} --save_file n_data_study200/exp_${ND}_trial_${I} --epochs 400 --scheduler 1
#python experiment_2D.py --n_data ${ND} --save_file hcp/exp_${ND}_trial_${I} --epochs 400 --scheduler 1 --display
python experiment_2D_reg.py --weight_decay 0.001 --n_data ${ND} --save_file n_data_study_reg001/exp_${ND}_trial_${I} --epochs 400 --scheduler 1 --display
