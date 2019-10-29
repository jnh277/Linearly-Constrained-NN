#!/bin/sh

# make executable with the following
# chmod u+x run_mag_data_tests.sh
# run using
# ./run_mag_data_tests.sh




for nt in 500 1000 1500 2000 2500 3000 5000
do
    python mag_data_experiment.py --scheduler 0 --epochs 1000 --n_train ${nt} --save_file mag_data_tests/sched_zero_ntrain_${nt}

done

for nt in 500 1000 1500 2000 2500 3000 5000
do
    python mag_data_experiment.py --scheduler 1 --epochs 1000 --save_file mag_data_tests/sched_one_ntrain_${nt} --n_train ${nt}

done


for nt in 500 1000 1500 2000 2500 3000 5000
do
    python mag_data_experiment.py --scheduler 1 --epochs 1000 --net_hidden_size 25 10 5 --save_file mag_data_tests/net25_ntrain_${nt} --n_train ${nt}

done


for nt in 500 1000 1500 2000 2500 3000 5000
do
    python mag_data_experiment.py --scheduler 1 --epochs 1000 --net_hidden_size 50 25 10 --save_file mag_data_tests/net50_ntrain_${nt} --n_train ${nt}

done

for nt in 250 500 1000 1500 2000 2500 5000
do
    python mag_data_experiment.py --scheduler 1 --epochs 1000 --net_hidden_size 100 50 25 --batch_size 250 --save_file mag_data_tests/net100_medbatch_ntrain_${nt} --n_train ${nt}

done

for nt in 250 500 1000 1500 2000 2500 5000
do
    python mag_data_experiment.py --scheduler 1 --epochs 1000 --net_hidden_size 50 25 10 --batch_size 250 --save_file mag_data_tests/net50_medbatch_ntrain_${nt} --n_train ${nt}

done

for nt in 250 500 1000 1500 2000 2500 5000
do
    python mag_data_experiment.py --scheduler 1 --epochs 1000 --net_hidden_size 25 10 5 --batch_size 100 --save_file mag_data_tests/net25_medbatch_ntrain_${nt} --n_train ${nt}

done

for nt in 250 500 1000 1500 2000 2500 5000
do
    python mag_data_experiment.py --scheduler 1 --epochs 1000 --net_hidden_size 100 50 25 --batch_size 100 --save_file mag_data_tests/net100_smallbatch_ntrain_${nt} --n_train ${nt}

done

for nt in 250 500 1000 1500 2000 2500 5000
do
    python mag_data_experiment.py --scheduler 1 --epochs 1000 --net_hidden_size 50 25 10 --batch_size 100 --save_file mag_data_tests/net50_smallbatch_ntrain_${nt} --n_train ${nt}

done

for nt in 250 500 1000 1500 2000 2500 5000
do
    python mag_data_experiment.py --scheduler 1 --epochs 1000 --net_hidden_size 25 10 5 --batch_size 100 --save_file mag_data_tests/net25_smallbatch_ntrain_${nt} --n_train ${nt}

done


