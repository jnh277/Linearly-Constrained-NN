# chmod u+x run_experiment_hcp_multi.sh
# run using ./run_experiment_hcp_multi.sh

#for nd in 100 200 300 400 500 1000 1500 2000 2500 3000 3500 4000
for nd in 100 200 500 1000 2000 3000
do
  for i in {1..25}
    do
        echo "Running with nd=$nd and i=$i"
        # runs python
       python experiment_2D_reg.py --weight_decay 0.001 --n_data ${nd} --save_file n_data_study_reg1/exp_${nd}_trial_${i} --epochs 400 --scheduler 1
    done
done