#!/bin/sh

# make executable with the following
# chmod u+x run_2D_net_size_study.sh
# run using ./run_2D_net_size_study.sh




# for sc in 1 2 4 6 8 10 12 14 16 18 20
# for sc in 2 4 6 8 10 12 14 16 18 20
do
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
    # for i in 1 2
    do
        z=`expr $sc \* 10`
        v=`expr $sc \* 5`
        python experiment_2D.py --n_data 4000 --net_hidden_size ${z} ${v} --save_file net_size_study/exp_${sc}_trial_${i} --epochs 400 --scheduler 1
    done
done