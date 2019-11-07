#!/bin/bash

# make executable with the following
# chmod u+x run_exp_hcp_netsize.sh
# run using ./run_exp_hcp_netsize.sh


for sc in 3 6 9 12 15 18 21
do
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
    do
        z=`expr $sc \* 10`
        v=`expr $sc \* 5`
         # calls each job script
        qsub -v Z=${z},V=${v},I=${i} run-experiment-vars-ns.sh
    done
done