#!/bin/sh

# make executable with the following
# chmod u+x run_2D_n_data_study.sh
# run using ./run_2D_n_data_study.sh


for nd in 100 200 300
do
    for i in 1 2 3
    do
        qsub –v ND=$nd,I=$i run-experiment-vars.sh      # calls each job script
    done
done