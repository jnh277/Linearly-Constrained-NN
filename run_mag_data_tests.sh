#!/bin/sh

# make executable with the following
# chmod u+x run_mag_data_tests.sh
# run using
# ./run_mag_data_tests.sh





python mag_data_experiment.py --scheduler 1 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test1 --net_hidden_size 200 100 50

python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test2 --net_hidden_size 200 100 50

python mag_data_experiment.py --scheduler 1 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test3 --net_hidden_size 200 100 50


python mag_data_experiment.py --scheduler 0 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test4 --net_hidden_size 200 100 50

python mag_data_experiment.py --scheduler 0 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test5 --net_hidden_size 200 100 50

python mag_data_experiment.py --scheduler 0 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test6 --net_hidden_size 200 100 50


python mag_data_experiment.py --scheduler 1 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test7 --net_hidden_size 100 50 25

python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test8 --net_hidden_size 100 50 25

python mag_data_experiment.py --scheduler 1 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test9 --net_hidden_size 100 50 25


python mag_data_experiment.py --scheduler 0 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test10 --net_hidden_size 100 50 25

python mag_data_experiment.py --scheduler 0 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test11 --net_hidden_size 100 50 25

python mag_data_experiment.py --scheduler 0 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test12 --net_hidden_size 100 50 25


python mag_data_experiment.py --scheduler 1 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test13 --net_hidden_size 50 25 10

python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test14 --net_hidden_size 50 25 10

python mag_data_experiment.py --scheduler 1 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test15 --net_hidden_size 50 25 10


python mag_data_experiment.py --scheduler 0 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test16 --net_hidden_size 50 25 10

python mag_data_experiment.py --scheduler 0 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test17 --net_hidden_size 50 25 10

python mag_data_experiment.py --scheduler 0 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test18 --net_hidden_size 50 25 10


python mag_data_experiment.py --scheduler 1 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test19 --net_hidden_size 25 10 5

python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test20 --net_hidden_size 25 10 5

python mag_data_experiment.py --scheduler 1 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test21 --net_hidden_size 25 10 5


python mag_data_experiment.py --scheduler 0 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test22 --net_hidden_size 25 10 5

python mag_data_experiment.py --scheduler 0 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test23 --net_hidden_size 25 10 5

python mag_data_experiment.py --scheduler 0 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test24 --net_hidden_size 25 10 5


python mag_data_experiment.py --scheduler 1 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test25 --net_hidden_size 100 50

python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test26 --net_hidden_size 100 50

python mag_data_experiment.py --scheduler 1 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test27 --net_hidden_size 100 50

python mag_data_experiment.py --scheduler 0 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test28 --net_hidden_size 100 50

python mag_data_experiment.py --scheduler 0 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test29 --net_hidden_size 100 50

python mag_data_experiment.py --scheduler 0 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test30 --net_hidden_size 100 50

python mag_data_experiment.py --scheduler 1 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test31 --net_hidden_size 50 25

python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test32 --net_hidden_size 50 25

python mag_data_experiment.py --scheduler 1 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test33 --net_hidden_size 50 25

python mag_data_experiment.py --scheduler 0 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test34 --net_hidden_size 50 25

python mag_data_experiment.py --scheduler 0 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test25 --net_hidden_size 50 25

python mag_data_experiment.py --scheduler 0 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test36 --net_hidden_size 50 25

python mag_data_experiment.py --scheduler 1 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test37 --net_hidden_size 150 75

python mag_data_experiment.py --scheduler 1 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test38 --net_hidden_size 150 75

python mag_data_experiment.py --scheduler 1 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test39 --net_hidden_size 150 75

python mag_data_experiment.py --scheduler 0 --batch_size 500 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test40 --net_hidden_size 150 75

python mag_data_experiment.py --scheduler 0 --batch_size 250 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test41 --net_hidden_size 150 75

python mag_data_experiment.py --scheduler 0 --batch_size 100 --epochs 1000 --n_train 5000 --save_file mag_data_tests2/test42 --net_hidden_size 150 75