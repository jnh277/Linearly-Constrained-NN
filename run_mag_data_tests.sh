#!/bin/sh

# make executable with the following
# chmod u+x run_mag_data_tests.sh
# run using
# ./run_mag_data_tests.sh



for nd in 250 500 1000 1500 2000 3000 4000 5000 6000
do
    for i in 1 2 3 4 5 6 7 8 9 10
    # for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
    # for i in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
    do
        python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 500 --n_train ${nd} --save_file mag_data_tests3/exp_${nd}_trial_${i} --net_hidden_size 150 75
    done
done
