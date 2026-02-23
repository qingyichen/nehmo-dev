import json

def collect_collision_initial_conditions(in_filename, out_filename):
    with open(in_filename, 'r') as f:
        data = json.load(f)
        collision_trials = data['inter_arm_collision_trials']
        initial_condigurations = {}
        
        for i in range(100):
            trial = collision_trials[i]
            initial_condigurations[str(i)] = {}
            initial_condigurations[str(i)]["qpos"] = data['intial_conditions'][str(trial)]["qpos"]
            initial_condigurations[str(i)]["qgoal"] = data['intial_conditions'][str(trial)]["qgoal"]

    with open(out_filename, 'w') as f:
        json.dump(initial_condigurations, f, indent=4)
        
if __name__ == "__main__":
    in_filename = "planning_traj/UR5_Kinova/simple_planner_500trials.json"
    out_filename = "planning_traj/UR5_Kinova/collision_initial_conditions.json"
    collect_collision_initial_conditions(in_filename, out_filename)