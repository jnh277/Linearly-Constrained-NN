#!/bin/bash

# make executable with the following
# chmod u+x run_dims_multi.sh
# run using ./run_dims_multi.sh


for d in 3 4 5 6 7 8 9 10
do
  for i in {1..50}
    do
         # calls each job script
        qsub -v D=${d},I=${i} run_dims_experiment.sh
    done
done