seeds=(0 1 2)
for seed in "${seeds[@]}";do
    python deepreach/experiment_scripts/train_hji_air3D.py --experiment_name experiment_IC --minWith zero --tMax 1.1 --velocity 0.75 --omega_max 3.0 --angle_alpha 1.2 --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --provide_initial_condition --seed $seed &
    python deepreach/experiment_scripts/train_hji_air3D.py --experiment_name experiment_original --minWith zero --tMax 1.1 --velocity 0.75 --omega_max 3.0 --angle_alpha 1.2 --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --seed $seed &
    python deepreach/experiment_scripts/train_hji_air3D.py --experiment_name experiment_symmetric_IC --minWith zero --tMax 1.1 --velocity 0.75 --omega_max 3.0 --angle_alpha 1.2 --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000 --use_symmetry --provide_initial_condition --seed $seed &
    wait
done

