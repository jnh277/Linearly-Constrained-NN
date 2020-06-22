#!/bin/bash

# make executable with the following
# chmod u+x run_pobs_multi.sh
# run using ./run_pobs_multi.sh


for w in 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.125 0.15
do
# for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
#    for i in 13 42
  for i in {1..200}
#  for i in {51..100}
    do
         # calls each job script
        qsub -v W=${w},I=${i} run_pobs_exp.sh
    done
done

