#!/bin/bash

# make executable with the following
# chmod u+x run_experiment_hcp_multi.sh
# run using ./run_experiment_hcp_multi.sh


# for nd in 250 500 1000 3000 6000 8000
for nd in 100 200 300 400 500 1000 1500 2000 2500 3000 3500 4000
do
# for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
#    for i in 13 42
  for i in {1..200}
#  for i in {51..100}
    do
         # calls each job script
        qsub -v ND=${nd},I=${i} run-experiment-vars.sh
    done
done

#        qsub -v ND=100,I=1 run-experiment-vars.sh
