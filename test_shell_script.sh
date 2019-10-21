#!/bin/sh

# make executable with the following
# chmod u+x test_shell_script.sh
# run using ./test_shell_script.sh

# python test_argparse.py

# python test_argparse.py --epochs 100

for nd in 100 200 300
do
    for i in 1 2
    do

        z=`expr $i \* 2`
        #echo $z
        python test_argparse.py --net_hidden_size ${z} ${z}
        # python test_argparse.py --save_file n_data_study/exp_${nd}_trial_${i}
    done
done