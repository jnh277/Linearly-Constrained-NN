#!/bin/sh

# make executable with the following
# chmod u+x run_2D_n_data_study.sh
# run using ./run_2D_n_data_study.sh


for nd in 100 200 300 400 500 1000 1500 2000 2500 3000 3500 4000
# for nd in 100
do
    # for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
    # for i in 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
    for i in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
    do
        python experiment_2D.py --n_data 2000 --save_file net_size_study/exp_${nd}_trial_${i} --epochs 400 --scheduler 1
    done
done