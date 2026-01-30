seeds=(0 1 2)
for seed in "${seeds[@]}";do
    python deepreach/experiment_scripts/train_hji_pointPlane.py --experiment_name experiment_IC  --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --provide_initial_condition --seed $seed &
    python deepreach/experiment_scripts/train_hji_pointPlane.py --experiment_name experiment_original --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --seed $seed &
    wait
done

