#!/bin/sh

# make executable with the following
# chmod u+x test_shell_script.sh
# run using ./test_shell_script.sh

# python test_argparse.py

# python test_argparse.py --epochs 100

for nd in 100 200 300 400 500 1000 1500 2000 2500 3000 3500 4000
do
    for i in {1..2}
    do
        python test_argparse.py --save_file exp_${nd}_trial_$i
    done
done