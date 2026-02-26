import os
import sys
import copy
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import zonopyrobots as robots2
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import deepreach.modules as modules
from UR5_datasets_and_training.training_utils import make_model
from envs.UR5_Kinova import DualArmEnv, FullStepRecorder, RobotStepViz
from agents.DualArm_agent import DualArmAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num_links', type=int, default=6)
    p.add_argument('--robot', type=str, default='UR5_Kinova')
    p.add_argument('--bc_model_dir_path', type=str, default='UR5_datasets_and_training')

    p.add_argument('--max_joint_velocity', type=float, default=0.5, required=False, help='Max joint velocity of the arm') # note: this param is dependent on the training setting of HJI

    p.add_argument('--provide_initial_condition', action='store_true', default=False, required=False, help='Whether to provide initial condition to the HJI model')
    p.add_argument('--use_symmetry', action='store_true', default=False, required=False, help='Whether the model considers symmetry during training.')
    p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

    p.add_argument('--logging_root', type=str, default='./hji_logs/logs_nn', help='root for logging hji training')
    p.add_argument('--experiment_name', type=str, default='experiment_IC_seed0')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # planning experiment settings
    # planner settings
    p.add_argument('--step_time', type=float, default=0.1, help='The step time of each plan.')
    p.add_argument('--safe_time', type=float, default=0.3, help='The duration of each plan to guarantee safety with HJI solution.')
    p.add_argument('--planner_mode', type=str, default='hji', help='Planner mode: hji, simple, or hji_simple.')
    p.add_argument('--buffer', type=float, default=0.05, help='The safety buffer for neural constraints.')
    # planning env settings
    p.add_argument('--num_trials', type=int, default=100, help='The number of random trials.')
    p.add_argument('--trials', nargs='+', help='The trial indices to run, will override the number trials argument', required=False)
    p.add_argument('--max_steps', type=int, default=200, help='The max number of steps allowed for a trial.')
    # rendering settings
    p.add_argument('--video', action='store_true', default=False, required=False, help='Whether to save planning video.')
    p.add_argument('--blender', action='store_true', default=False, required=False, help='Whether to save planning blender file.')
    p.add_argument('--traj', action='store_true', default=False, required=False, help='Whether to add trajetcory history to the blender file.')
    p.add_argument('--traj_spacing', type=int, default=8, help='The spacing between each sampled trajectory history')

    # results saving
    p.add_argument('--save_stats', action='store_true', default=False, required=False, help='Whether to save planning statistics.')
    p.add_argument('--save_traj', action='store_true', default=False, required=False, help='Whether to save planned trajectory.')
    
    opt = p.parse_args()
    return  opt

def load_robot(opt):
    robots2.DEBUG_VIZ = False
    robot = robots2.ZonoArmRobot.load(os.path.join(os.getcwd(), f'envs/arm_urdfs/ur5_kinova/ur5_kinova_gen3.urdf'), create_joint_occupancy=False)
    return robot
    
def load_model(opt, experiment_name=None):
    # Setting to plot
    if experiment_name is None:
        experiment_name = opt.experiment_name
    ckpt_path = f'{opt.logging_root}_{opt.robot}/{experiment_name}/checkpoints/model_final.pth'
    activation = 'sine'
    
    # Load the boundary condition model first
    bc_model = None
    if opt.provide_initial_condition:
        with open(os.path.join(opt.bc_model_dir_path, 'bc_models', opt.robot + '_training_setting.json'), 'r') as f:
            json_config = json.load(f)
            params = copy.deepcopy(opt)
            params.__dict__.update(json_config)
        bc_model, _ = make_model(params)

    # Load HJI model
    model = modules.NN_SymmetryICNet(in_features=opt.num_links*2+1, out_features=1, type=activation, mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3,
                             bc_model=bc_model, 
                             max_joint_velocity=opt.max_joint_velocity, num_links=opt.num_links,
                             provide_initial_condition=opt.provide_initial_condition, 
                             use_symmetry=opt.use_symmetry)
    model.to(device=opt.device)
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def compute_plan(planner, qpos, qgoal, step_time=0.1, safe_time=0.3, timestep_discretization=10, buffer=0.0, planner_mode='hji'):
    # map qpos to [qpos1, qpos2] first
    env_to_qpos_mapping = np.array([0, 3, 4, 7, 9, 10, 1, 2, 5, 6, 8, 11], dtype=np.int32)
    qpos_to_env_mapping = np.array([0, 6, 7, 1, 2, 8, 9, 3, 10, 4, 5, 11], dtype=np.int32)
    env_qpos = qpos
    qpos = env_qpos[env_to_qpos_mapping]
    qgoal = qgoal[env_to_qpos_mapping]
    
    num_links = qpos.shape[0] // 2
    arm1_state = qpos[:num_links]
    arm2_state = qpos[num_links:]
    arm1_goal = qgoal[:num_links]
    arm2_goal = qgoal[num_links:]
    
    t0 = time.time()
    arm1_action = planner.plan(agent_state=arm1_state,
                            other_agent_state=arm2_state,
                            goal_state=arm1_goal,
                            step_time=step_time, 
                            safe_time=safe_time,
                            buffer=buffer,
                            planner_mode=planner_mode).cpu().numpy()
    
    t1 = time.time()
    arm2_action = planner.plan(agent_state=arm2_state,
                            other_agent_state=arm1_state,
                            goal_state=arm2_goal,
                            step_time=step_time, 
                            safe_time=safe_time,
                            buffer=buffer,
                            planner_mode='simple').cpu().numpy()
    t2 = time.time()
    
    
    velocity_action = np.concatenate((arm1_action, arm2_action))
    new_qpos = qpos.copy() + velocity_action * step_time
    velocity_action = velocity_action[qpos_to_env_mapping]
    velocity_action = np.tile(velocity_action, timestep_discretization).reshape(timestep_discretization, -1)
    configuration = env_qpos + np.linspace(0., step_time, num=timestep_discretization, endpoint=True).reshape(-1,1) * velocity_action
    return configuration, velocity_action, [t1-t0], qpos, qgoal, new_qpos
    

if __name__ == '__main__':
    opt = parse_args()
    video = opt.video
    blender = opt.blender
    timestep_discretization = 10
    planner_name = opt.planner_mode
    if opt.planner_mode == 'hji':
        planner_name += f'_{opt.experiment_name}'
        planner_name += f'_safeT{opt.safe_time}_b{opt.buffer}'
    planner_name += '_planner'
    
    # load collision-only initial conditions
    with open(os.path.join('envs', 'collision_initial_conditions.json'), 'r') as f:
        collision_initial_conditions = json.load(f)
    qpos_to_env_mapping = np.array([0, 6, 7, 1, 2, 8, 9, 3, 10, 4, 5, 11], dtype=np.int32)
    
    # planning statistics
    success_trials = []
    collision_trials = []
    self_collision_trials = []
    inter_arm_collision_trials = []
    planning_times = []
    num_steps_for_success_trials = []
    initial_conditions_and_num_steps_taken = {}
    
    # rendering preparations
    if video:
        import platform
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        video_folder = f'planning_videos/{opt.robot}/{planner_name}'
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
    if blender:
        blender_folder = f'planning_blenders/{opt.robot}/{planner_name}'
        if not os.path.exists(blender_folder):
            os.makedirs(blender_folder)
    
    # planning preparations
    robot = load_robot(opt=opt)
    planner = DualArmAgent(
        max_joint_velocity=0.5,
        num_links=6, 
        val_func_model=None if opt.planner_mode=='simple' else load_model(opt=opt),
        val_func_model2=None if (opt.planner_mode=='simple' or opt.planner_mode=='hji_simple') else load_model(opt=opt, experiment_name="experiment_symmetric_IC_Kinova_seed1"),
        device=opt.device,
    )
    
    # run planning experiments
    trials = range(opt.num_trials)
    if opt.trials is not None:
        trials = [int(i) for i in opt.trials]
    for i_trial in tqdm(trials):
        planned_trajectoy = []
        env = DualArmEnv(robot=robot.urdf,
                            t_step=opt.step_time,
                            timestep_discretization=10,
                            step_type='direct',
                            check_self_collision=True,
                            verbose_self_collision=True,
                            renderer='pyrender-offscreen' if not opt.blender else 'blender', # or 'pyrender', or 'blender'
                            seed=i_trial)
        env.reset(
            qpos=np.array(collision_initial_conditions[str(i_trial)]['qpos'])[qpos_to_env_mapping],
            qgoal=np.array(collision_initial_conditions[str(i_trial)]['qgoal'])[qpos_to_env_mapping]
        )
        if blender:
            renderer_args = {'filename': os.path.join(blender_folder, f"{planner_name}_random_scene{i_trial}.blend")}
            env.renderer_kwargs = renderer_args
        if video:
            video_path = os.path.join(video_folder, f'video{i_trial}.mp4')
            video_recorder = FullStepRecorder(env, path=video_path)
        if opt.traj:
            traj_viz = RobotStepViz(save_every_n_iteration=opt.traj_spacing)
            env.add_render_callback('trajectory', traj_viz.render_callback, needs_time=False)

        for i_step in range(opt.max_steps):
            configuration, velocity, t, start_qpos, qgoal, new_qpos = compute_plan(planner=planner,
                                                   qpos=env.qpos,
                                                   qgoal=env.qgoal, 
                                                   step_time=opt.step_time,
                                                   safe_time=opt.safe_time,
                                                   timestep_discretization=10,
                                                   buffer=opt.buffer,
                                                   planner_mode=opt.planner_mode)
            
            if i_step == 0:
                initial_conditions_and_num_steps_taken[i_trial] = {
                    'qpos': start_qpos.tolist(), 
                    'qgoal': qgoal.tolist()
                }
            if opt.save_traj:
                planned_trajectoy.append(new_qpos.tolist())

            planning_times += t
            observation, reward, done, info = env.step((configuration, velocity))
            if video:
                video_recorder.capture_frame()
            if blender:
                env.render()
            success, collision = info['success'], info['collision_info']['in_collision']
            if collision:
                collision_trials.append(i_trial)
                if info['collision_info']['inter_arm_collision']:
                    inter_arm_collision_trials.append(i_trial)
                if info['collision_info']['self_collision']:
                    self_collision_trials.append(i_trial)
                break
            if success:
                success_trials.append(i_trial)
                num_steps_for_success_trials.append(i_step + 1)
                break
        initial_conditions_and_num_steps_taken[i_trial].update({
            'num_steps_taken': i_step + 1,
            'success': success, 
            'collision': collision,
            'trajectory': planned_trajectoy,
        })
        
        if video:
            video_recorder.close()
        if blender:
            env.close()
    
    # save statistics
    if opt.save_stats:
        stats = {
            'num_successes': len(success_trials),
            'num_collisions': len(collision_trials),
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
            'inter_arm_collision_trials': inter_arm_collision_trials,
            'self_collision_trials': self_collision_trials,
            'planner_settings': opt.__dict__,
            'intial_conditions': initial_conditions_and_num_steps_taken
        }
        if opt.save_traj:
            stats_folder = f'planning_traj/{opt.robot}'
        else:
            stats_folder = f'planning_results/{opt.robot}'
        if not os.path.exists(stats_folder):
            os.makedirs(stats_folder)
        with open(os.path.join(stats_folder, planner_name + f'_{len(trials)}trials' + '.json'), 'w') as f:
            json.dump(stats, f, indent=4)
    