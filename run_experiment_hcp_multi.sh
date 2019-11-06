#!/bin/sh

# make executable with the following
# chmod u+x run_experiment_hcp_multi.sh
# run using ./run_experiment_hcp_multi.sh


for nd in 100 200 300
do
    for i in 1 2 3
    do
        qsub –v ND=$nd,I=$i ./Linearly-Constrained-NN/run-experiment-vars.sh      # calls each job script
    done
done