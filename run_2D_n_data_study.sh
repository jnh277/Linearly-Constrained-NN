#!/bin/sh

# make executable with the following
# chmod u+x run_2D_n_data_study.sh
# run using ./run_2D_n_data_study.sh


# for nd in 100 200 300 400 500 1000 1500 2000 2500 3000 3500 4000
for nd in 100 2000
do
    for i in 1 2
    do
        python experiment_2D.py --n_data $nd --save_file n_data_study/exp_${nd}_trial_$i
    done
done