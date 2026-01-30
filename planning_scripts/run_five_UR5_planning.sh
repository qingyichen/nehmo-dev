n_trials=100
video=""

python planning/FiveArm_planning.py --num_trials $n_trials --buffer 0.05 --safe_time 0.3 --save_stats --provide_initial_condition --use_symmetry --experiment_name experiment_symmetric_IC_seed1 $video

