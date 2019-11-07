#!/bin/bash

# make executable with the following
# chmod u+x run_experiment_hcp_multi.sh
# run using ./run_experiment_hcp_multi.sh


for nd in 250 500 1000 3000 6000
do
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
    do
         # calls each job script
        qsub -v ND=${nd},I=${i} run-experiment-vars.sh
    done
done
