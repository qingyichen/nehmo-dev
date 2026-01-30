import os
import sys
import copy
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import deepreach.modules as modules
from envs.PointPlane_env import PointPlaneEnv
from agents.PointPlane_agent import PointPlaneAgent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num_agents', type=int, default=2)
    
    p.add_argument('--provide_initial_condition', action='store_true', default=False, required=False, help='Whether to provide initial condition')
    p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

    p.add_argument('--logging_root', type=str, default='./hji_logs/logs_PointPlane', help='root for logging hji training')
    p.add_argument('--experiment_name', type=str, default='experiment_time_invariance_seed0')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # planning experiment settings
    # planner settings
    p.add_argument('--step_time', type=float, default=0.1, help='The step time of each plan.')
    p.add_argument('--safe_time', type=float, default=0.3, help='The duration of each plan to guarantee safety with HJI solution.')
    p.add_argument('--planner_mode', type=str, default='hji', help='Planner mode: hji, simple, or adversary.')
    p.add_argument('--buffer', type=float, default=0.01, help='The safety buffer for neural constraints.')
    # planning env settings
    p.add_argument('--num_trials', type=int, default=100, help='The number of random trials.')
    p.add_argument('--trials', nargs='+', help='The trial indices to run, will override the number trials argument', required=False)
    p.add_argument('--max_steps', type=int, default=200, help='The max number of steps allowed for a trial.')
    # rendering settings
    p.add_argument('--video', action='store_true', default=False, required=False, help='Whether to save planning video.')
    p.add_argument('--refined', action='store_true', default=False, required=False, help='Whether to save planning video in refined quality.')
    # results saving
    p.add_argument('--save_stats', action='store_true', default=False, required=False, help='Whether to save planning statistics.')
    p.add_argument('--save_traj', action='store_true', default=False, required=False, help='Whether to save planned trajectory.')

    opt = p.parse_args()
    return  opt
    
def load_model(opt):
    # Setting to plot
    ckpt_path = f'{opt.logging_root}/{opt.experiment_name}/checkpoints/model_final.pth'
    activation = 'sine'
    # Load HJI model
    model = modules.PointPlane_ICNet(in_features=5, out_features=1, type=activation, mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3,
                             provide_initial_condition=opt.provide_initial_condition, collisionR=0.2)
    model.to(device=opt.device)
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def compute_plan(planner, agent_states, goals, buffer=1e-3, planner_mode='hji'):
    num_agents = agent_states.shape[0]
    planning_times = []
    agent_actions = []
    for i in range(num_agents):
        other_agent_states = torch.cat([agent_states[0:i,:], agent_states[i+1:,:]], dim=0)
        t0 = time.time()
        agent_action = planner.plan(agent_state=agent_states[i],
                                other_agent_states=other_agent_states,
                                goal_state=goals[i],
                                buffer=buffer,
                                planner_mode=planner_mode)
        torch.cuda.synchronize()
        t1 = time.time()
        planning_times.append(t1 - t0)
        agent_actions.append(agent_action.cpu())

    action = torch.vstack(agent_actions)
    return action, planning_times, agent_states, goals
    

if __name__ == '__main__':
    opt = parse_args()
    video = opt.video
    planner_name = opt.planner_mode
    if opt.planner_mode == 'hji':
        planner_name += f'_{opt.experiment_name}'
        planner_name += f'_safeT{opt.safe_time}_b{opt.buffer}'
    planner_name += '_planner'
    opt.max_steps = opt.num_agents * 20
    
    # planning statistics
    success_trials = []
    collision_trials = []
    planning_times = []
    num_steps_for_success_trials = []
    plan_length_per_agent = []
    succ_path_lengths = []
    initial_conditions_and_num_steps_taken = {}
    
    # planning preparations
    planner = PointPlaneAgent(
        velocity_limit=0.5,
        step_time=0.1,
        safe_time=0.3,
        val_func_model=None if opt.planner_mode=='simple' else load_model(opt=opt),
        device=opt.device,
    )
    
    # run planning experiments
    trials = range(opt.num_trials)
    if opt.trials is not None:
        trials = [int(i) for i in opt.trials]
    for i_trial in tqdm(trials):
        planned_trajectoy = []
        env = PointPlaneEnv(seed=i_trial,
                       num_random_agents=opt.num_agents,
                       device=opt.device
                    )
        last_states = None
        finish = False
        path_length = 0.
        for i_step in range(opt.max_steps):
            action, t, start_states, goal_states = compute_plan(planner=planner,
                                                   agent_states=env.agent_states, 
                                                   goals=env.goal_states,
                                                   buffer=opt.buffer,
                                                   planner_mode=opt.planner_mode)
            current_states = env.agent_states.clone()
            if i_step == 0:
                initial_conditions_and_num_steps_taken[i_trial] = {
                    'agent_start_states': start_states.cpu().tolist(), 
                    'goal': goal_states.cpu().tolist()
                }
            if opt.save_traj:
                planned_trajectoy.append(current_states.tolist())
            planning_times += t
            observation, reward, done, info = env.step(action)
            success, collision = info['success'], info['collision']
            if collision:
                collision_trials.append(i_trial)
            if success:
                success_trials.append(i_trial)
                num_steps_for_success_trials.append(i_step + 1)
                path_length += torch.sum(torch.linalg.norm(action * opt.step_time, dim=-1)).item()
                succ_path_lengths.append(path_length / opt.num_agents)
            finish = success or collision or (i_step == opt.max_steps-1) 

            if last_states is not None:
                if torch.all(torch.linalg.norm(last_states - current_states, dim=1) < 1e-2):
                    finish = True
            if video:
                env.render(close=finish, save=True, save_name=f'trial{i_trial}', show=False, refined=opt.refined)
            if finish:
                break
            last_states = current_states
            path_length += torch.sum(torch.linalg.norm(action * opt.step_time, dim=-1)).item()
            
        initial_conditions_and_num_steps_taken[i_trial].update({
            'num_steps_taken': i_step + 1,
            'success': success, 
            'collision': collision,
            'trajectory': planned_trajectoy,
        })

    # save statistics
    if opt.save_stats:
        stats = {
            'num_successes': len(success_trials),
            'num_collisions': len(collision_trials),
            'path_length_each_agent_for_succ_trials': {
                'mean': np.mean(succ_path_lengths),
                'std': np.std(succ_path_lengths),
            },
            'planning_times': {
                'mean': np.mean(planning_times),
                'std': np.std(planning_times),
            },
            'num_steps_for_success_trials': {
                'mean': np.mean(num_steps_for_success_trials),
                'std': np.std(num_steps_for_success_trials)
            },
            'success_trials': success_trials,
            'collision_trials': collision_trials,
            'planner_settings': opt.__dict__,
            'intial_conditions': initial_conditions_and_num_steps_taken
        }
        if opt.save_traj:
            stats_folder = f'planning_traj/PointPlane_{opt.num_agents}agents'
        else:
            stats_folder = f'planning_results/PointPlane_{opt.num_agents}agents'
        if not os.path.exists(stats_folder):
            os.makedirs(stats_folder)
        with open(os.path.join(stats_folder, planner_name + f'_{len(trials)}trials' + '.json'), 'w') as f:
            json.dump(stats, f, indent=4)
    