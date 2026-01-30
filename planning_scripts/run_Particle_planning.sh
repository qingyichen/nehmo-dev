n_trials=100
video=""
num_agentss=(2 8 16 32)

for num_agents in "${num_agentss[@]}";do
    python planning/PointPlane_planning.py --num_trials $n_trials --safe_time 0.3 --save_stats  --provide_initial_condition --experiment_name experiment_IC_seed0 $video --num_agents $num_agents 
    python planning/PointPlane_planning.py --num_trials $n_trials --planner_mode simple --save_stats $video --num_agents $num_agents 
    python planning/PointPlane_planning.py --num_trials $n_trials --safe_time 0.3 --save_stats --experiment_name experiment_original_seed0 $video --num_agents $num_agents 
done