seeds=(0 1 2)
for seed in "${seeds[@]}";do
    python deepreach/experiment_scripts/train_hji_nn.py --experiment_name experiment_IC --minWith zero --tMax 0.6 --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --provide_initial_condition --seed $seed &
    python deepreach/experiment_scripts/train_hji_nn.py --experiment_name experiment_original --minWith zero --tMax 0.6 --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --seed $seed &
    python deepreach/experiment_scripts/train_hji_nn.py --experiment_name experiment_symmetric_IC --minWith zero --tMax 0.6 --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --provide_initial_condition --use_symmetry --seed $seed &
    wait
done
