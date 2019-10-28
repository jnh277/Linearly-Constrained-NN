#!/bin/sh

# make executable with the following
# chmod u+x run_2D_net_size_study.sh
# run using ./run_2D_net_size_study.sh




# for sc in 1 2 4 6 8 10 12 14 16 18 20
# for sc in 2 4 6 8 10 12 14 16 18 20
for sc in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    #for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
    for i in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
    do
        # z=`expr $sc \* 10`
        # v=`expr $sc \* 5`
        z=`expr $sc \* 2`
        v=`expr $sc \* 1`
        python experiment_2D.py --n_data 4000 --net_hidden_size ${z} ${v} --save_file net_size_study2/exp_${z}_trial_${i} --epochs 400 --scheduler 1
    done
done