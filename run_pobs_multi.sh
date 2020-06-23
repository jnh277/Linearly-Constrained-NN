#!/bin/bash

# make executable with the following
# chmod u+x run_pobs_multi.sh
# run using ./run_pobs_multi.sh


#for w in 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.75 1.0 1.5 2.0 5.0 10.0

for w in 0 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.2 2.4 4.4 8.8 16.0 32.0 64.0 100.0
do
for i in {1..50}
#  for i in {1..200}
#  for i in {51..100}
    do
         # calls each job script
        qsub -v W=${w},I=${i} run_pobs_exp.sh
    done
done

