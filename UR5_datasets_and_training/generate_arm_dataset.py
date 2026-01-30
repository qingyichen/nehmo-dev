import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from envs.DualArm_env import DualArmEnv
import zonopyrobots as robots2
import numpy as np
from tqdm import tqdm
import torch
import pickle

def collect_inter_arm_distance(num_sub_datasets=80, num_data=32000):
    for i_dataset in range(num_sub_datasets):
        env = DualArmEnv(robot=robot.urdf,
                            timestep_discretization=1,
                            step_type='direct',
                            check_self_collision=True,
                            verbose_self_collision=True,
                            seed=i_dataset)
        filename = f'seed{i_dataset}num_data{num_data}.pkl'
        subdataset_qpos = np.zeros((num_data, 12))
        subdataset_distances = np.zeros(num_data)

        for i in tqdm(range(num_data)):
            distance = env.distance_sampler(consider_self_collision=check_self_collision)
            env_to_qpos_mapping = np.array([0, 3, 4, 7, 9, 10, 1, 2, 5, 6, 8, 11], dtype=np.int32)
            qpos = env.qpos[env_to_qpos_mapping]
            subdataset_qpos[i] = qpos.copy()
            subdataset_distances[i] = distance
            
            
        with open(os.path.join(dataset_dir, filename), 'wb') as f:
            data = {
                'qpos': torch.from_numpy(subdataset_qpos).float(),
                'distances': torch.from_numpy(subdataset_distances).float()
            }
            pickle.dump(data, f)
    print(f"{num_sub_datasets} sub-datasets with each having {num_data} generated.")
    
def collect_self_collision_arm_distance(num_sub_datasets=80, num_data=32000):
    for i_dataset in range(num_sub_datasets):
        env = DualArmEnv(robot=robot.urdf,
                            timestep_discretization=1,
                            step_type='direct',
                            check_self_collision=True,
                            verbose_self_collision=True,
                            seed=i_dataset,
                            viz_goal=False)
        filename = f'seed{i_dataset}num_data{num_data}.pkl'
        subdataset_qpos = np.zeros((num_data, 6))
        subdataset_distances = np.zeros(num_data)

        for i in tqdm(range(num_data)):
            distance = env.self_collision_distance_sampler()
            env_to_qpos_mapping = np.array([0, 3, 4, 7, 9, 10, 1, 2, 5, 6, 8, 11], dtype=np.int32)
            qpos = env.qpos[env_to_qpos_mapping][:6]
            subdataset_qpos[i] = qpos.copy()
            subdataset_distances[i] = distance
            
        with open(os.path.join(dataset_dir, filename), 'wb') as f:
            data = {
                'qpos': torch.from_numpy(subdataset_qpos).float(),
                'distances': torch.from_numpy(subdataset_distances).float()
            }
            pickle.dump(data, f)
    print(f"{num_sub_datasets} sub-datasets with each having {num_data} generated.")
    
def self_collision_test():
    video = False
    blender = True
    renderer = 'pyrender'
    if video:
        renderer = 'pyrender-offscreen'
    if blender:
        renderer = 'blender'
    
    env = DualArmEnv(robot=robot.urdf,
                            timestep_discretization=5,
                            step_type='direct',
                            renderer=renderer,
                            check_self_collision=True,
                            verbose_self_collision=True,
                            viz_goal=True)
    qpos = np.zeros(12)
    qpos[0:2] = 3.14

    env.reset(qpos=qpos)
    env_to_qpos_mapping = np.array([0, 3, 4, 7, 9, 10, 1, 2, 5, 6, 8, 11], dtype=np.int32)
    qpos_to_env_mapping = np.array([0, 6, 7, 1, 2, 8, 9, 3, 10, 4, 5, 11], dtype=np.int32)
    step_time = 0.1
    timestep_discretization = 5
    

    if video:
        sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
        from envs.DualArm_env import FullStepRecorder
        import platform
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        video_folder = f'planning_videos/sc'
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        video_path = os.path.join(video_folder, f'video_sc_test.mp4')
        video_recorder = FullStepRecorder(env, path=video_path)
    
    num_steps =40
    for i in range(num_steps):
        env_qpos = env.qpos
        print(env_qpos)
        arm1_action = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        arm2_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        velocity_action = np.concatenate((arm1_action, arm2_action))[qpos_to_env_mapping]
        velocity_action = np.tile(velocity_action, timestep_discretization).reshape(timestep_discretization, -1)
        configuration = env_qpos + np.linspace(0., step_time, num=timestep_discretization, endpoint=True).reshape(-1,1) * velocity_action
        
        observation, reward, done, info = env.step((configuration, velocity_action))
        if video:
            video_recorder.capture_frame()
        if blender:
            env.render()
        if info['collision_info']['self_collision']:
            print("SELF COLLISION!")
        if info['collision_info']['in_collision'] and not info['collision_info']['self_collision']:
            print("INTERARM COLLISION")
    if video:
            video_recorder.close()
    if blender:
        env.close() 
    
    

if __name__ == '__main__':
    robots2.DEBUG_VIZ = False
    robot = robots2.ZonoArmRobot.load(os.path.join(os.getcwd(),'envs/arm_urdfs/ur5/dual_ur5.urdf'), create_joint_occupancy=False)

    num_sub_datasets = 80
    check_self_collision = False
    num_data = 32000
    
    dataset_dir = f"UR5_datasets_and_training/UR5_d0.6_mesh_distances_{'sc_' if check_self_collision else ''}dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    collect_inter_arm_distance()
    
    
    
    

    
    