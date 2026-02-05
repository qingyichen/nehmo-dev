# TODO DOCUMNET & CLEAN

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from urchin import URDF
from typing import Union, Tuple, List

import numpy as np
import trimesh
import pyrender
from collections import OrderedDict
import zonopyrobots as robots2
import copy

from enum import Enum
class STEP_TYPE(str, Enum):
    DIRECT = 'direct'
    INTEGRATION = 'integration'

class RENDERER(str, Enum):
    PYRENDER = 'pyrender'
    PYRENDER_OFFSCREEN = 'pyrender-offscreen'
    BLENDER = 'blender'
    # MATPLOTLIB = 'matplotlib' # disabled for now

# A base environment for basic kinematic simulation of any URDF loaded by Urchin without any reward
class DualArmEnv:
    '''Base class for kinematic simulation of any URDF loaded by Urchin without any reward.'''
    DIMENSION = 3

    def __init__(self,
            robot: Union[URDF, str] = 'Kinova3', # robot model
            max_episode_steps: int = 300,
            step_type: STEP_TYPE = 'direct',
            t_step: float = 0.5,
            timestep_discretization: int = 50,       # these are values per t_step
            integration_discretization: int = 100,   # these are values per t_step, if step type is direct, this is ignored.
            # GOAL related
            goal_threshold = 0.1, # goal threshold
            goal_use_mesh = True, # use mesh for goal,
            viz_goal=True,
            # COLLISION related
            check_collision: bool = True,
            check_self_collision: bool = True,
            verbose_self_collision: bool = False,
            use_bb_collision: bool = False,
            collision_discretization: int = 10,     # these are values per t_step, if self_collision is False, this is ignored.
            collision_use_visual: bool = False,
            collision_links: List[str] = None,
            collision_simplification_faces: int = 500,
            check_joint_limits: bool = True,
            enforce_joint_pos_limit: bool = True,
            verbose_joint_limits: bool = False,
            renderer: RENDERER = 'pyrender',
            render_frames: int = 30,
            render_fps: int = 60,
            render_size: Tuple[int, int] = (600, 600),
            render_mesh: bool = True,
            renderer_kwargs: dict = {},
            reopen_on_close: bool = False,
            seed: int = 0,
            ):
        '''Initialize the base URDF environment

        Args:
            robot: Union[URDF, str]
                The URDF robot model to use in the environment.
            max_episode_steps: int
                The maximum number of steps in an episode.
            step_type: STEP_TYPE
                The type of step to take in the environment.
                Either 'direct' or 'integration'.
                If 'direct', the action should be a tuple of qpos and qvel for each timestep_discretization.
                If 'integration', the action should be an array of qdd which is integrated to get qpos and qvel.
            t_step: float
                The time step for each step in the environment.
            timestep_discretization: int
                The number of timestep steps to take for each action in the environment.
                t_step / timestep_discretization would be the time step for each saved substep in the environment.
            integration_discretization: int
                The number of integration steps to take for each action in the environment.
                t_step / integration_discretization would be the time step for each integrated substep in the environment.
                This is only used if step_type is 'integration'.
            check_collision: bool
                Whether to check for collisions in the environment.
            check_self_collision: bool
                Whether to check for self collisions in the environment.
            verbose_self_collision: bool
                Whether to provide verbose information about self collisions in the environment.
                This will provide the pairs of links that are in collision at each timestep and the time of the collision.
            use_bb_collision: bool
                Whether to use bounding box collision checking.
                If True, the collision checking will only use the bounding box of the mesh.
                If False, the collision checking will first use the bounding box and then the full mesh.
            collision_discretization: int
                The number of collision steps to take for each action in the environment.
                Positions will be interpolated to this number of steps for collision checking.
            collision_use_visual: bool
                Whether to use the visual meshes for collision checking.
                If True, the collision checking will use the visual meshes of the robot.
                If False, the collision checking will use the collision meshes of the robot.
            collision_links: List[str]
                The links to check for collisions.
                If None, all links will be checked for collisions.
            collision_simplification_faces: int
                The number of faces to simplify the collision mesh to.
                This reduces the expense of mesh collision checking.
            check_joint_limits: bool
                Whether to check for joint limits in the environment.
                Information about violated joint limits will be stored in the info dictionary.
            enforce_joint_pos_limit: bool
                Whether to enforce the joint position limits in the environment.
            verbose_joint_limits: bool
                Whether to provide verbose information about joint limits in the environment.
                This will provide the joints that are exceeded at each timestep and the time of the exceedance.
            renderer: RENDERER
                The renderer to use for the environment.
            render_frames: int
                The number of frames to render for a single step in the environment.
            render_fps: int
                The desired frames per second for rendering.
            render_size: Tuple[int, int]
                The size of the render.
            render_mesh: bool
                Whether to render the mesh.
            renderer_kwargs: dict
                Extra kwargs to pass to the renderer.
            reopen_on_close: bool
                Whether to reopen the renderer if it is closed.
                This only applies to the pyrender renderer.
            seed: int
                The seed for the random number generator used in the environment.
        '''
        
        ## Gym metadata compat
        self.metadata = {
            'render_fps': render_fps
        }

        # Setup the random number generator
        self.np_random = np.random.default_rng(seed=seed)

        ### Robot Specs
        # Load the robot & get important properties
        if isinstance(robot, str):
            robot = URDF.load(robot)
        self.robot = robot
        self.robot_graph = self.robot._G.to_undirected()
        
        self.dof = len(self.robot.actuated_joints)
        self.continuous_joints = []
        pos_lim = [[-np.Inf, np.Inf]]*self.dof
        vel_lim = [np.Inf]*self.dof
        eff_lim = [np.Inf]*self.dof
        for i,joint in enumerate(self.robot.actuated_joints):
            if joint.joint_type == 'continuous':
                self.continuous_joints.append(i)
            elif joint.joint_type in ['floating', 'planar']:
                raise NotImplementedError
            if joint.limit is not None:
                lower = joint.limit.lower if joint.limit.lower is not None else -np.Inf
                upper = joint.limit.upper if joint.limit.upper is not None else np.Inf
                pos_lim[i] = [lower, upper]
                vel_lim[i] = joint.limit.velocity
                eff_lim[i] = joint.limit.effort
        self.pos_lim = np.array(pos_lim).T 
        self.vel_lim = np.array(vel_lim)
        self.eff_lim = np.array(eff_lim) # Unused for now

        # vars
        # State variables for the configuration of the URDF robot
        self.state = np.zeros(self.dof*2)
        self.init_state = np.zeros(self.dof*2)
        self.last_trajectory = self.qpos

        ### Simulation parameters for each step of this environment
        if step_type not in list(STEP_TYPE):
            raise ValueError
        self.step_type = step_type
        self._max_episode_steps = max_episode_steps
        self.t_step = t_step
        self.timestep_discretization = timestep_discretization
        self.integration_discretization = integration_discretization
        self.done = False
        # vars
        self._elapsed_steps = 0
        self.terminal_observation = None
        
        # Goal
        self.goal_threshold = goal_threshold
        self.goal_use_mesh = goal_use_mesh

        ### Collision checking parameters
        self.check_collision = check_collision
        self.check_self_collision = check_self_collision
        self.verbose_self_collision = verbose_self_collision
        self.use_bb_collision = use_bb_collision
        self.collision_discretization = collision_discretization
        self.collision_simplification_faces = collision_simplification_faces
        self.collision_use_visual = collision_use_visual
        if collision_links is None:
            collision_links = [link.name for link in self.robot.links]
        self.collision_links = set(collision_links)

        
        ### NEW
        self.robot1_collision_manager = trimesh.collision.CollisionManager()
        self.robot2_collision_manager = trimesh.collision.CollisionManager()
        self.robot_collision_objs = set()
        for link in self.robot.links:
            if link.name not in self.collision_links:
                 continue
            if not self.collision_use_visual and link.collision_mesh is not None:
                mesh = link.collision_mesh
                if link.name[0] == '1':
                    self.robot1_collision_manager.add_object(link.name, mesh=mesh)
                elif link.name[0] == 'k':
                    self.robot2_collision_manager.add_object(link.name, mesh=mesh)
            self.robot_collision_objs.add(link.name)
        # vars
        self.collision_info = None

        ### Joint limit check parameters
        self.check_joint_limits = check_joint_limits
        self.enforce_joint_pos_limit = enforce_joint_pos_limit
        self.verbose_joint_limits = verbose_joint_limits
        self.joint_limit_info = None

        ### Visualization Parameters
        if renderer not in list(RENDERER):
            raise ValueError
        self.renderer = renderer
        self.render_frames = render_frames
        self.render_fps = render_fps
        self.render_size = render_size
        self.render_mesh = render_mesh
        self.renderer_kwargs = renderer_kwargs
        self.reopen_on_close = reopen_on_close
        self.render_callbacks = {}
        self.viz_goal = viz_goal
        # vars
        self.scene = None
        self.scene_map = None
        self.scene_viewer = None
        self._info = None
        self.last_clear_calls = []
    
    def reset(self, qpos: np.ndarray = None, qvel: np.ndarray = 0, state: np.ndarray = None, qgoal: np.ndarray = None):
        '''Reset the environment to a given state or a random state.
        
        Args:
            qpos: np.ndarray, shape (dof,)
                The position of each joint in the URDF robot.
            qvel: np.ndarray, shape (dof,)
                The velocity of each joint in the URDF robot.
            state: np.ndarray, shape (dof*2,)
                The state of the URDF robot, where the first dof elements are the qpos and the next dof elements are the qvel.
            qgoal: np.ndarray, Optional
                Goal position for the robot.
                If not provided, then a random goal position will be generated.
        '''
        # set or generate a state
        if state is not None:
            self.state = np.copy(state)
            self.qpos = self._wrap_cont_joints(self.qpos)
        else:
            if qpos is not None:
                self.qpos = self._wrap_cont_joints(qpos)
            else:
                # Generate a random position for each of the joints
                # Try 100 times until there is no self collision
                self.qpos = self._generate_free_configuration(n_tries=100)
            
            # Set the initial velocity to 0 or whatever is provided
            self.qvel = qvel
                        
        # Generate the goal position
        if qgoal is not None:
            self.qgoal = self._wrap_cont_joints(qgoal)
        else:
            # Generate a random position for each of the joints
            # Try 10 times until there is no self collision
            self.qgoal = self._generate_free_configuration(n_tries=10)
            
        # self.qpos = self.qpos * 0.0 + 0.15
        # self.qgoal = self.qpos.copy()
        # self.qgoal *= -1
        # # self.qgoal[np.array([0,3,4,7,9,10])] *= -1
        # self.qgoal[0] += 3.14
        # self.qgoal[1] += 3.14
        # self.qgoal[5] += 3.14
        # self.qgoal[8] += 3.14
        # self.qgoal[2] *= -1
            
    
        self.init_state = np.copy(self.state)
        self.done = False
        self._elapsed_steps = 0
        self.last_trajectory = self.qpos
        self.terminal_observation = None
        self.collision_info = None
        self.joint_limit_info = None
        self._info = None
        self.initial_qpos = self.qpos.copy()
        
        # Create a new file if using blender renderer
        if self.renderer == RENDERER.BLENDER:
            import bpy
            if self.scene is not None:
                self.close()
            bpy.ops.wm.read_factory_settings(use_empty=True)

        return self.get_observations()
    
    def self_collision_distance_sampler(self, qpos=None):
        pos_lim = np.copy(self.pos_lim)
        pos_lim /= 2.0 # Note: scale the joint range from 2pi to pi. This is peculiar to UR5 robot.
        if qpos is None:
            self.qpos = self.np_random.uniform(low=pos_lim[0], high=pos_lim[1])
        else:
            self.qpos = qpos
        fk_dict = self.robot.link_fk(self.qpos, use_names=True)
        
        # NOTE: we only need to cosider one robot for self collision
        temp_fk_dict = {}
        for name, transform in fk_dict.items():
            if name in self.robot_collision_objs:
                if name[0] == '1':
                    temp_fk_dict[name] = transform
                    self.robot1_collision_manager.set_transform(name, transform)
        fk_dict = temp_fk_dict
        min_distance = float('inf')
        min_distance_pair = None
        
        collision_manager = self.robot1_collision_manager
        for link_name, transform in fk_dict.items():
            # for every link in the collision manager
            sinle_link_collision_manager = trimesh.collision.CollisionManager()
            link_mesh = self.robot.link_map[link_name].collision_mesh
            removed_name_mesh_transforms = [(link_name, link_mesh, fk_dict[link_name])]
            sinle_link_collision_manager.add_object(link_name, link_mesh, transform=fk_dict[link_name])
            # we remove its adjacent links and itself, and then compute the min distances
            collision_manager.remove_object(link_name)
            for other_link in self.robot_graph[self.robot.link_map[link_name]]:
                if other_link.name == 'world_base':
                    continue
                other_link_name = other_link.name
                other_link_mesh = other_link.collision_mesh
                removed_name_mesh_transforms.append((other_link_name, other_link_mesh, fk_dict[other_link_name]))
                collision_manager.remove_object(other_link_name)

            min_distance_with_other, name_pair = collision_manager.min_distance_other(sinle_link_collision_manager, return_names=True)
            if min_distance_with_other < min_distance:
                min_distance = min_distance_with_other
                min_distance_pair = name_pair
            # clean up: add back the link and its adjacencies
            for other_link_name, other_link_mesh, other_link_transform in removed_name_mesh_transforms:
                collision_manager.add_object(other_link_name, other_link_mesh, other_link_transform)
            sinle_link_collision_manager.remove_object(link_name)
        return min_distance
        
        
    def distance_sampler(self, consider_self_collision=False, qpos=None):
        pos_lim = np.copy(self.pos_lim)
        pos_lim[pos_lim == np.Inf] = 2.0 * np.pi 
        pos_lim[pos_lim == -np.Inf] = -2.0 * np.pi
        pos_lim /= 2.0 # Note: scale the joint range from 2pi to pi. This is peculiar to UR5 robot.
        if qpos is None:
            self.qpos = self.np_random.uniform(low=pos_lim[0], high=pos_lim[1])
        else:
            self.qpos = qpos
        fk_dict = self.robot.link_fk(self.qpos, use_names=True)
        
        distance1, name_pairs = self._different_arm_collision_check(fk_dict=fk_dict, distance=True)
        distance2 = float('inf')
        if consider_self_collision:
            distance2, name_pairs = self._single_arm_self_collision_distance_check(fk_dict=fk_dict, robot_index='1')
        return min(distance1, distance2)
        
    def _generate_free_configuration(self, n_tries=100):
        '''Generate a random configuration for the robot that is not in self collision.
        
        Args:
            n_tries: int
                The number of times to try to generate a free configuration.
        
        Returns:
            np.ndarray, shape (dof,)
                The free configuration of the robot.
            
        Raises:
            RuntimeError: If a free configuration cannot be generated in n_tries attempts.
        '''
        # to match the config distribution of baseline methods
        pos_lower_lim = np.array([-5, -1.0, 1.0, -0.2, -2.5, -0.0001] + [-np.pi, -2.4, -np.pi, -2.66, -np.pi, -2.23])
        pos_upper_lim = np.array([-1.5, 0.5, 2.0, 0.8, -0.5, 0.0] + [np.pi, 2.4, np.pi, 2.66, np.pi, 2.23])
        qpos_to_env_mapping = np.array([0, 6, 7, 1, 2, 8, 9, 3, 10, 4, 5, 11], dtype=np.int32)
        pos_lower_lim = pos_lower_lim[qpos_to_env_mapping]
        pos_upper_lim = pos_upper_lim[qpos_to_env_mapping]
        pos_lim = np.stack((pos_lower_lim, pos_upper_lim))
        
        # temporarily disable verbose self collisions
        original_verbosity = self.verbose_self_collision
        self.verbose_self_collision = False

        def pos_helper():
            new_pos = self.np_random.uniform(low=pos_lim[0], high=pos_lim[1]) # NOTE: divide by 2 for UR5
            new_pos = self._wrap_cont_joints(new_pos)
            fk_dict = self.robot.link_fk(new_pos, use_names=True)

            collision, _ = self._self_collision_check(fk_dict, use_bb=self.use_bb_collision)
            inter_collision, _ = self._different_arm_collision_check(fk_dict, distance=False)
            collision = collision or inter_collision
            return collision, new_pos
        
        for _ in range(n_tries):
            collision, new_pos = pos_helper()
            if not collision:
                # restore verbosity
                self.verbose_self_collision = original_verbosity
                return new_pos
        raise RuntimeError

    def get_observations(self):
        '''Get a deepcopy of the observations of the environment.'''
        observation = {
            'qpos': self.qpos,
            'qvel': self.qvel,
        }
        return copy.deepcopy(observation)
    
    def step(self, action):
        '''Perform a step in the environment given an action.
        
        Args:
            action: np.ndarray
                The action to take in the environment.
                If the step type is direct, this should be a tuple of qpos and qvel, each of shape (timestep_discretization, dof).
                If the step type is integration, this should be an array of shape (dof) representing the qdd.
        '''
        if self.step_type == STEP_TYPE.DIRECT:
            q = np.copy(action[0])
            qd = np.copy(action[1])
            if len(q) != self.timestep_discretization or len(qd) != self.timestep_discretization:
                raise RuntimeError
        elif self.step_type == STEP_TYPE.INTEGRATION:
            qdd = self._interpolate_q(action, self.integration_discretization)
            t = self.t_step / self.integration_discretization
            qd_delt = np.cumsum(qdd * t, axis=0)
            qd = self.qvel + qd_delt
            q = self.qpos + np.cumsum(0.5 * qdd * t * t, axis=0)
            q[1:] += np.cumsum(qd[:-1] * t, axis=0)
        # Perform joint limit check if desired before enforcing joint limits
        if self.check_joint_limits:
            self.joint_limit_info = self._joint_limit_check(q, qd)
        if self.enforce_joint_pos_limit:
            q, qd = self._enforce_joint_pos_limit(q, qd)
        # Perform collision checks
        if self.check_collision:
            self.collision_info = self._collision_check_internal(q)
        # Get the timestep discretization of each
        q = self._interpolate_q(q, self.timestep_discretization)
        qd = self._interpolate_q(qd, self.timestep_discretization)
        self.last_trajectory = q
        # wraparound the positions for continuous joints (do this after checks becuase checks do different discretizations)
        q = self._wrap_cont_joints(q)
        # store the final state
        self.qpos = q[-1]
        self.qvel = qd[-1]
        # Update extra variables and return
        self._elapsed_steps += 1
        observations = self.get_observations()
        self._info = self.get_info()
        reward = self.get_reward(action)
        self.done = self.done if self._elapsed_steps < self._max_episode_steps else True
        self._info['success'] = reward == 1
        return observations, reward, self.done, self._info

    def get_reward(self, action):
        '''Get the reward for the current step given the action. Must be implemented by the subclass if rewards are desired.
        
        Args:
            action: np.ndarray
                The action taken in the environment.
        '''
        collision = self.info['collision_info']['in_collision']
        dist = np.linalg.norm(self._wrap_cont_joints(self.qpos - self.qgoal))
        success = (dist < self.goal_threshold) and (not collision)

        self.done = self.done or success or collision
        
        return int(success)
        
    def get_info(self):
        info = {
            'init_qpos': self.init_qpos,
            'init_qvel': self.init_qvel,
            'last_trajectory': self.last_trajectory
            }
        if self.check_collision:
            info['collision_info'] = self.collision_info
        if self.check_joint_limits:
            info['joint_limit_exceeded'] = self.joint_limit_info
        if self.done:
            if self.terminal_observation is None:
                self.terminal_observation = copy.deepcopy(self.get_observations())
            info['terminal_observation'] = self.terminal_observation
        return copy.deepcopy(info)
    
    def _different_arm_collision_check(self, fk_dict, distance=True):
        for name, transform in fk_dict.items():
            if name in self.robot_collision_objs:
                if name[0] == '1':
                    self.robot1_collision_manager.set_transform(name, transform)
                elif name[0] == 'k':
                    self.robot2_collision_manager.set_transform(name, transform)
                elif  name != 'world_base':
                    raise RuntimeError
        if not distance:
            collision, name_pairs = self.robot1_collision_manager.in_collision_other(self.robot2_collision_manager, return_names=True)
            return collision, name_pairs
        else:
            distance, name_pairs = self.robot1_collision_manager.min_distance_other(self.robot2_collision_manager, return_names=True)
            return distance, name_pairs
        
    def _single_arm_self_collision_distance_check(self, fk_dict, robot_index='1', distance=True):
        for name, transform in fk_dict.items():
            if name in self.robot_collision_objs:
                if name[0] == robot_index:
                    self.robot1_collision_manager.set_transform(name, transform)
        min_distance = float('inf')
        true_names = set()
        for collision_manager in [self.robot1_collision_manager]:
            distance, name_pairs = collision_manager.min_distance_internal(return_names=True)
            if distance <= 0.:
                n1, n2 = name_pairs
                link1 = self.robot.link_map[n1]
                link2 = self.robot.link_map[n2]

                if link2 not in self.robot_graph[link1]:
                    if self.verbose_self_collision:
                        true_names.add((n1, n2))
                        min_distance = min(distance, min_distance)
                    else:
                        min_distance = min(distance, min_distance)
                return min_distance, true_names
        return min_distance, true_names
        
    
    def _self_collision_check(self, fk_dict, use_bb=False):
        '''Check for self collisions given a forward kinematics dictionary.

        Args:
            fk_dict: dict
                A dictionary containing the forward kinematics of each link in the robot.
            use_bb: bool
                Whether to use the bounding box for collision checking.

        Returns:
            bool
                Whether there is a self collision.
            set
                The pairs of links that are in self collision.
        '''     
        for name, transform in fk_dict.items():
            if name in self.robot_collision_objs:
                if name[0] == '1':
                    self.robot1_collision_manager.set_transform(name, transform)
                elif name[0] == 'k':
                    self.robot2_collision_manager.set_transform(name, transform)
                else:
                    if name != 'world_base':
                        raise RuntimeError(f"Unknown affliation of {name}")
                
        for collision_manager in [self.robot1_collision_manager, self.robot2_collision_manager]:
            collision, name_pairs = collision_manager.in_collision_internal(return_names=True)
            # if we have collision, make sure it isn't pairwise
            if collision:
                # check all name pairs
                true_names = set()
                for n1, n2 in name_pairs:
                    link1 = self.robot.link_map[n1]
                    link2 = self.robot.link_map[n2]
                    # If the links aren't neighbors in the underlying robot graph, there's
                    # a true collision
                    if link2 not in self.robot_graph[link1]:
                        if self.verbose_self_collision:
                            true_names.add((n1, n2))
                        else:
                            return True, set()
                return len(true_names) > 0, true_names
        return False, set()
        
    def _collision_check_internal(self, q_step):
        '''Check for collisions at each step of the given q_step.
        
        Args:
            q_step: np.ndarray, shape (timestep_discretization, dof)
                The joint positions to check for collisions.
        
        Returns:
            dict
                A dictionary containing the collision information.
        '''
        out = {
            'in_collision': False,
            'inter_arm_collision': False,
        }
        # Interpolate the given for q for the number of steps we want
        q_check = self._interpolate_q(q_step, self.collision_discretization)
        batch_fk_dict = self.robot.link_fk_batch(cfgs=q_check, links=self.robot_collision_objs)
        merge = self.collision_check(batch_fk_dict)
        out.update(merge)
        # Check each of the configurations
        self_collision = False
        out['self_collision'] = self_collision
        self_collision_pairs = {}
        inter_arm_collision_pairs = {}
        for i in range(self.collision_discretization):
            # Comprehend a name: transform from the batch link fk dictionary
            fk_dict = {link.name: pose[i] for link, pose in batch_fk_dict.items()}
            if self.check_self_collision:
                self_collision, pairs = self._self_collision_check(fk_dict, use_bb=False)
            inter_arm_collision, interarm_pairs = self._different_arm_collision_check(fk_dict, distance=False)
            collision = self_collision or inter_arm_collision
            if self.verbose_self_collision and collision:
                collision_time = float(i)/self.collision_discretization * self.t_step
                if self_collision:
                    self_collision_pairs[collision_time] = pairs
                if inter_arm_collision:
                    inter_arm_collision_pairs[collision_time] = interarm_pairs
            elif collision:
                out['in_collision'] = True
                out['inter_arm_collision'] = inter_arm_collision
                out['self_collision'] = self_collision
                break
        if len(self_collision_pairs) > 0 or len(inter_arm_collision_pairs) > 0:
            out['in_collision'] = True
            out['inter_arm_collision'] = inter_arm_collision
            out['self_collision'] = self_collision
            out['inter_arm_collision_pairs'] = inter_arm_collision_pairs
            out['self_collision_pairs'] = self_collision_pairs
                    
        # Return as a dict
        return out
    
    def collision_check(self, batch_fk_dict):
        '''Check for collisions given a batch of forward kinematics.
        Empty function to be implemented by the subclass.

        Args:
            batch_fk_dict: dict
                A dictionary containing the forward kinematics of each link in the robot.
        
        Returns:
            dict
                A dictionary containing the collision information.
        '''
        out = {}
        return out

    def render(self,
               render_frames = None,
               render_fps = None,
               render_size = None,
               render_mesh = None,
               renderer = None,
               renderer_kwargs = None):
        '''Render the last step taken in the environment.
        
        Args:
            render_frames (Optional[int]):
                The number of frames to render.
            render_fps (Optional[int]):
                The frames per second to render.
            render_size (Optional[Tuple[int, int]]):
                The size of the render.
            render_mesh (Optional[bool]):
                Whether to render the mesh.
            renderer (Optional[RENDERER]):
                The renderer to use.
            renderer_kwargs (Optional[dict]):
                Extra kwargs to pass to the renderer.
        '''
        render_frames = render_frames if render_frames is not None else self.render_frames
        render_fps = render_fps if render_fps is not None else self.render_fps
        render_size = render_size if render_size is not None else self.render_size
        render_mesh = render_mesh if render_mesh is not None else self.render_mesh
        renderer = renderer if renderer is not None else self.renderer
        renderer_kwargs = renderer_kwargs if renderer_kwargs is not None else self.renderer_kwargs

        # Prep the q to render
        q_render = self._interpolate_q(self.last_trajectory, render_frames)
        q_render = self._wrap_cont_joints(q_render)
        fk = self.robot.visual_trimesh_fk_batch(q_render)
        
        # Clear any prior calls
        [clear() for clear in self.last_clear_calls]
        self.last_clear_calls = []

        # Call any onetime render callbacks and make list of callbacks to call each timestep
        step_callback_list = []
        full_callback_list = []
        for callback, needs_time in self.render_callbacks.values():
            if needs_time:
                step_callback_list.append(callback)
            else:
                full_callback_list.append(callback)

        # Create the render callback for each timestep
        def step_render_callback(timestep):
            cleanup_list = []
            for callback in step_callback_list:
                cleanup = callback(self, timestep)
                if cleanup is not None:
                    cleanup_list.append(cleanup)
            return lambda: [cleanup() for cleanup in cleanup_list]
        
        def full_render_callback():
            cleanup_list = []
            for callback in full_callback_list:
                cleanup = callback(self)
                if cleanup is not None:
                    cleanup_list.append(cleanup)
            return lambda: [cleanup() for cleanup in cleanup_list]

        # Generate the render
        if renderer == RENDERER.PYRENDER:
            import time
            if self.scene is None:
                self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
            self.last_clear_calls.append(full_render_callback())
            step_cleanup_callback = None
            for i in range(render_frames):
                proc_time_start = time.time()
                if not self.scene_viewer.is_active and self.reopen_on_close:
                    self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
                elif not self.scene_viewer.is_active:
                    return
                self.scene_viewer.render_lock.release()
                self.scene_viewer.render_lock.acquire()
                for mesh, node in self.scene_map.items():
                    pose = fk[mesh][i]
                    node.matrix = pose
                timestep = float(i+1)/render_frames * self.t_step
                if step_cleanup_callback is not None:
                    step_cleanup_callback()
                step_cleanup_callback = step_render_callback(timestep)
                self.scene_viewer.render_lock.release()
                if render_fps is not None:
                    proc_time = time.time() - proc_time_start
                    pause_time = max(0, 1.0/render_fps - proc_time)
                    time.sleep(pause_time)
                self.scene_viewer.render_lock.acquire()
            if step_cleanup_callback is not None:
                self.last_clear_calls.append(step_cleanup_callback)
        elif renderer == RENDERER.PYRENDER_OFFSCREEN:
            if self.scene is None:
                self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
            self.last_clear_calls.append(full_render_callback())
            step_cleanup_callback = None
            color_frames = [None]*render_frames
            depth_frames = [None]*render_frames
            for i in range(render_frames):
                for mesh, node in self.scene_map.items():
                    pose = fk[mesh][i]
                    node.matrix = pose
                timestep = float(i+1)/render_frames * self.t_step
                if step_cleanup_callback is not None:
                    step_cleanup_callback()
                step_cleanup_callback = step_render_callback(timestep)
                color_frames[i], depth_frames[i] = self.scene_viewer.render(self.scene)
            if step_cleanup_callback is not None:
                self.last_clear_calls.append(step_cleanup_callback)
            return color_frames, depth_frames

    def add_render_callback(self, key, callback, needs_time = False):
        '''Add a render callback to the environment.
        
        Render callbacks are called either once per render or for each render timestep.
        The callback should take the environment as an argument and return a cleanup function.
        If the cleanup function is not None, it will be called after the render is complete.
        To remove a render callback, use the remove_render_callback method.
        If the callback needs the current time (it's called once per timestep), set needs_time to True.

        Args:
            key: str
                The key to identify the callback.
            callback: function
                The callback to call.
            needs_time (Optional[bool]):
                Whether the callback needs the current time. Default is False.
        '''
        self.render_callbacks[key] = (callback, needs_time)
    
    def remove_render_callback(self, key):
        '''Remove a render callback from the environment.

        Args:
            key: str
                The key of the callback to remove.
        '''
        self.render_callbacks.pop(key, None)

    def spin(self, wait_for_enter=False):
        '''Spin the environment to render the last step taken.
        This is mostly for debugging purposes when using the pyrender renderer.

        Args:
            wait_for_enter (Optional[bool]):
                Whether to wait for the enter key to be pressed in order to continue.
                Otherwise, waits for the scene viewer to close.
        '''
        if self.renderer == RENDERER.PYRENDER:
            if self.scene_viewer is not None and self.scene_viewer.is_active:
                if wait_for_enter:
                    self.scene_viewer.render_lock.release()
                    input("Waiting for enter to continue...")
                    self.scene_viewer.render_lock.acquire()
                else:
                    self.scene_viewer.render_lock.release()
                    while self.scene_viewer.is_active:
                        import time
                        time.sleep(0.1)
        
    def close(self):
        '''Close the environment and cleanup any resources.'''
        if self.renderer == RENDERER.PYRENDER:
            if self.scene_viewer is not None and self.scene_viewer.is_active:
                self.scene_viewer.close_external()
        elif self.renderer == RENDERER.PYRENDER_OFFSCREEN:
            if self.scene_viewer is not None:
                self.scene_viewer.delete()
        elif self.renderer == RENDERER.BLENDER:
            import bpy
            if self.scene is not None:
                final_frame = self.scene.frame_current - 1
                self.scene.frame_end = final_frame
                self.scene.frame_set(1)
                bpy.ops.wm.save_mainfile(compress=True)
        self.last_clear_calls = []
        self.scene = None
        self.scene_map = None
        self.scene_viewer = None

    def _wrap_cont_joints(self, pos: np.ndarray) -> np.ndarray:
        '''Wrap the continuous joints to be within the range [-pi, pi].

        Args:
            pos: np.ndarray, shape (dof,)
                The joint positions to wrap.
        
        Returns:
            np.ndarray, shape (dof,)
                The wrapped joint positions.
        '''
        pos = np.copy(pos)
        pos[..., self.continuous_joints] = (pos[..., self.continuous_joints] + np.pi) % (2 * np.pi) - np.pi
        return pos

    # Utility to interpolate the position of the joints to a desired number of steps
    def _interpolate_q(self, q: np.ndarray, out_steps, match_end=True, match_start=False):
        '''Interpolate the joint positions to a desired number of steps.

        Args:
            q: np.ndarray, shape (timestep_discretization, dof)
                The joint positions to interpolate.
            out_steps: int
                The number of steps to interpolate to.
            match_end: bool
                Whether to match the end of the trajectory time.
            match_start: bool
                Whether to match the start of the trajectory time.
        
        Returns:
            np.ndarray, shape (out_steps, dof)
                The interpolated joint positions.
        
        Raises:
            ValueError: If neither match_start nor match_end is True.
        '''
        if not (match_start or match_end):
            raise ValueError
        if len(q.shape) == 1:
            q = np.expand_dims(q,0)
        in_times = np.linspace(0, 0.5, num=len(q) + int(not match_start), endpoint=match_end)
        coll_times = np.linspace(0, 0.5, num=out_steps + int(not match_start), endpoint=match_end)
        if not match_start:
            in_times = in_times[1:]
            coll_times = coll_times[1:]
        q_interpolated = [None]*self.dof
        for i in range(self.dof):
            q_interpolated[i] = np.interp(coll_times, in_times, q[:,i])
        q_interpolated = np.array(q_interpolated, order='F').T # Use fortran order so it's C order when transposed.
        return q_interpolated

    def _joint_limit_check(self, q_step, qd_step):
        '''Check if position or velocity limits are exceeded.
        
        Args:
            q_step: np.ndarray, shape (timestep_discretization, dof)
                The joint positions to check.
            qd_step: np.ndarray, shape (timestep_discretization, dof)
                The joint velocities to check.
        
        Returns:
            dict
                A dictionary containing the joint limit information.
        '''
        pos_exceeded = np.logical_or(q_step < self.pos_lim[0], q_step > self.pos_lim[1])
        vel_exceeded = np.abs(qd_step) > self.vel_lim
        exceeded = np.any(pos_exceeded) or np.any(vel_exceeded)
        out = {'exceeded': exceeded}
        if exceeded and self.verbose_joint_limits:
            def nonzero_to_odict(times, joints):
                n_times = len(q_step)
                times = times.astype(float) / n_times * self.t_step
                exceeded_dict = OrderedDict()
                for t, j in zip(times, joints):
                    joint_list = exceeded_dict.get(t,[])
                    joint_list.append(j)
                    exceeded_dict[t] = joint_list
                return exceeded_dict
            times, joints = np.nonzero(pos_exceeded)
            out['pos_exceeded'] = nonzero_to_odict(times, joints)
            times, joints = np.nonzero(vel_exceeded)
            out['vel_exceeded'] = nonzero_to_odict(times, joints)
        return out
    
    def _enforce_joint_pos_limit(self, q_step, qd_step):
        '''Enforce the joint position limits kinematically.

        Args:
            q_step: np.ndarray, shape (timestep_discretization, dof)
                The joint positions to enforce the limits on.
            qd_step: np.ndarray, shape (timestep_discretization, dof)
                The joint velocities to enforce the limits on.

        Returns:
            np.ndarray, shape (timestep_discretization, dof)
                The joint positions with the limits enforced.
            np.ndarray, shape (timestep_discretization, dof)
                The joint velocities with the limits enforced.
        '''
        q_step = np.copy(q_step)
        qd_step = np.copy(qd_step)
        # Lazy enforce lower
        pos_exceeded_lower = q_step < self.pos_lim[0]
        q_step[pos_exceeded_lower] = np.broadcast_to(self.pos_lim[0], q_step.shape)[pos_exceeded_lower]
        qd_step[pos_exceeded_lower] = 0
        # Lazy enforce upper
        pos_exceeded_upper = q_step > self.pos_lim[1]
        q_step[pos_exceeded_upper] = np.broadcast_to(self.pos_lim[1], q_step.shape)[pos_exceeded_upper]
        qd_step[pos_exceeded_upper] = 0
        return q_step, qd_step
                    
    def render(self,
               render_frames = None,
               render_fps = None,
               render_size = None,
               render_mesh = None,
               renderer = None,
               renderer_kwargs = None):
        '''Render the last step taken in the environment.
        
        Args:
            render_frames (Optional[int]):
                The number of frames to render.
            render_fps (Optional[int]):
                The frames per second to render.
            render_size (Optional[Tuple[int, int]]):
                The size of the render.
            render_mesh (Optional[bool]):
                Whether to render the mesh.
            renderer (Optional[RENDERER]):
                The renderer to use.
            renderer_kwargs (Optional[dict]):
                Extra kwargs to pass to the renderer.
        '''
        render_frames = render_frames if render_frames is not None else self.render_frames
        render_fps = render_fps if render_fps is not None else self.render_fps
        render_size = render_size if render_size is not None else self.render_size
        render_mesh = render_mesh if render_mesh is not None else self.render_mesh
        renderer = renderer if renderer is not None else self.renderer
        renderer_kwargs = renderer_kwargs if renderer_kwargs is not None else self.renderer_kwargs

        # Prep the q to render
        q_render = self._interpolate_q(self.last_trajectory, render_frames)
        q_render = self._wrap_cont_joints(q_render)
        fk = self.robot.visual_trimesh_fk_batch(q_render)
        
        # Clear any prior calls
        [clear() for clear in self.last_clear_calls]
        self.last_clear_calls = []

        # Call any onetime render callbacks and make list of callbacks to call each timestep
        step_callback_list = []
        full_callback_list = []
        for callback, needs_time in self.render_callbacks.values():
            if needs_time:
                step_callback_list.append(callback)
            else:
                full_callback_list.append(callback)

        # Create the render callback for each timestep
        def step_render_callback(timestep):
            cleanup_list = []
            for callback in step_callback_list:
                cleanup = callback(self, timestep)
                if cleanup is not None:
                    cleanup_list.append(cleanup)
            return lambda: [cleanup() for cleanup in cleanup_list]
        
        def full_render_callback():
            cleanup_list = []
            for callback in full_callback_list:
                cleanup = callback(self)
                if cleanup is not None:
                    cleanup_list.append(cleanup)
            return lambda: [cleanup() for cleanup in cleanup_list]

        # Generate the render
        if renderer == RENDERER.PYRENDER:
            import time
            if self.scene is None:
                self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
            self.last_clear_calls.append(full_render_callback())
            step_cleanup_callback = None
            for i in range(render_frames):
                proc_time_start = time.time()
                if not self.scene_viewer.is_active and self.reopen_on_close:
                    self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
                elif not self.scene_viewer.is_active:
                    return
                self.scene_viewer.render_lock.release()
                self.scene_viewer.render_lock.acquire()
                for mesh, node in self.scene_map.items():
                    pose = fk[mesh][i]
                    node.matrix = pose
                timestep = float(i+1)/render_frames * self.t_step
                if step_cleanup_callback is not None:
                    step_cleanup_callback()
                step_cleanup_callback = step_render_callback(timestep)
                self.scene_viewer.render_lock.release()
                if render_fps is not None:
                    proc_time = time.time() - proc_time_start
                    pause_time = max(0, 1.0/render_fps - proc_time)
                    time.sleep(pause_time)
                self.scene_viewer.render_lock.acquire()
            if step_cleanup_callback is not None:
                self.last_clear_calls.append(step_cleanup_callback)
        elif renderer == RENDERER.PYRENDER_OFFSCREEN:
            if self.scene is None:
                self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
            self.last_clear_calls.append(full_render_callback())
            step_cleanup_callback = None
            color_frames = [None]*render_frames
            depth_frames = [None]*render_frames
            for i in range(render_frames):
                for mesh, node in self.scene_map.items():
                    pose = fk[mesh][i]
                    node.matrix = pose
                timestep = float(i+1)/render_frames * self.t_step
                if step_cleanup_callback is not None:
                    step_cleanup_callback()
                step_cleanup_callback = step_render_callback(timestep)
                color_frames[i], depth_frames[i] = self.scene_viewer.render(self.scene)
            if step_cleanup_callback is not None:
                self.last_clear_calls.append(step_cleanup_callback)
            return color_frames, depth_frames
        # Blender rendering
        elif renderer == RENDERER.BLENDER:
            if self.scene is None:
                self._create_blender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
            self.last_clear_calls.append(full_render_callback())
            step_cleanup_callback = None
            for i in range(render_frames):
                current_frame = self.scene.frame_current
                for mesh, bpy_obj in self.scene_map.items():
                    pose = fk[mesh][i]
                    bpy_obj.location = pose[:3,3]
                    bpy_obj.rotation_quaternion = trimesh.transformations.quaternion_from_matrix(pose)
                    bpy_obj.keyframe_insert(data_path="location", frame=current_frame)
                    bpy_obj.keyframe_insert(data_path="rotation_quaternion", frame=current_frame)
                timestep = float(i+1)/render_frames * self.t_step
                if step_cleanup_callback is not None:
                    step_cleanup_callback()
                step_cleanup_callback = step_render_callback(timestep)
                self.scene.frame_set(current_frame + 1)
            if step_cleanup_callback is not None:
                self.last_clear_calls.append(step_cleanup_callback)
            # Checkpoint save without outputting to stdout
            save_interval = self.renderer_kwargs.get('save_interval', 5)
            if save_interval is not None and self._elapsed_steps % save_interval == 0:
                import bpy
                import io
                from contextlib import redirect_stdout
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    final_frame = self.scene.frame_current - 1
                    self.scene.frame_end = final_frame
                    bpy.ops.wm.save_mainfile(compress=True)

    def add_render_callback(self, key, callback, needs_time = False):
        '''Add a render callback to the environment.
        
        Render callbacks are called either once per render or for each render timestep.
        The callback should take the environment as an argument and return a cleanup function.
        If the cleanup function is not None, it will be called after the render is complete.
        To remove a render callback, use the remove_render_callback method.
        If the callback needs the current time (it's called once per timestep), set needs_time to True.

        Args:
            key: str
                The key to identify the callback.
            callback: function
                The callback to call.
            needs_time (Optional[bool]):
                Whether the callback needs the current time. Default is False.
        '''
        self.render_callbacks[key] = (callback, needs_time)
    
    def remove_render_callback(self, key):
        '''Remove a render callback from the environment.

        Args:
            key: str
                The key of the callback to remove.
        '''
        self.render_callbacks.pop(key, None)

    def spin(self, wait_for_enter=False):
        '''Spin the environment to render the last step taken.
        This is mostly for debugging purposes when using the pyrender renderer.

        Args:
            wait_for_enter (Optional[bool]):
                Whether to wait for the enter key to be pressed in order to continue.
                Otherwise, waits for the scene viewer to close.
        '''
        if self.renderer == RENDERER.PYRENDER:
            if self.scene_viewer is not None and self.scene_viewer.is_active:
                if wait_for_enter:
                    self.scene_viewer.render_lock.release()
                    input("Waiting for enter to continue...")
                    self.scene_viewer.render_lock.acquire()
                else:
                    self.scene_viewer.render_lock.release()
                    while self.scene_viewer.is_active:
                        import time
                        time.sleep(0.1)
        
    def close(self):
        '''Close the environment and cleanup any resources.'''
        if self.renderer == RENDERER.PYRENDER:
            if self.scene_viewer is not None and self.scene_viewer.is_active:
                self.scene_viewer.close_external()
        elif self.renderer == RENDERER.PYRENDER_OFFSCREEN:
            if self.scene_viewer is not None:
                self.scene_viewer.delete()
        elif self.renderer == RENDERER.BLENDER:
            import bpy
            if self.scene is not None:
                final_frame = self.scene.frame_current - 1
                self.scene.frame_end = final_frame
                self.scene.frame_set(1)
                bpy.ops.wm.save_mainfile(compress=True)
        self.last_clear_calls = []
        self.scene = None
        self.scene_map = None
        self.scene_viewer = None
        
    def _create_blender_scene(self, render_mesh, render_size, render_fps, renderer_kwargs):
        '''Create a base blender scene for rendering.

        Args:
            render_mesh: bool
                Whether to render the mesh.
            render_size: Tuple[int, int]
                The size of the render in the scene.
            render_fps: int
                The frames per second to save in the blend metadata.
            renderer_kwargs: dict
                Extra kwargs to pass to the renderer.
        '''
        import bpy
        filename = renderer_kwargs.get('filename', 'blender_render.blend') # default filename
        filename = get_next_available_filename(os.path.join(os.getcwd(), filename))
        print("Saving blend file to: ", filename)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.save_as_mainfile(filepath=filename)
        bpy.context.preferences.filepaths.save_version = 0
        self.scene = bpy.data.scenes.get('Scene')
        self.scene.render.fps = render_fps
        self.scene.render.fps_base = 1.0
        self.scene.render.resolution_x = render_size[0]
        self.scene.render.resolution_y = render_size[1]
        self.scene.render.resolution_percentage = 100
        # Create the robot
        # OK for now
        mat = bpy.data.materials.new(name="RobotMaterial")
        ### Pre-set Color
        mat.blend_method = 'OPAQUE'
        # mat.shadow_method = 'NONE'
        mat.show_transparent_back = False
        mat.use_nodes = True
        robot_principled = mat.node_tree.nodes["Principled BSDF"]
        # hex_value = ''
        # b = (hex_value & 0xFF) / 255.0
        # g = ((hex_value >> 8) & 0xFF) / 255.0
        # r = ((hex_value >> 16) & 0xFF) / 255.0
        robot_principled.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        robot_principled.inputs["Alpha"].default_value = 1 #
        ###
        robot_collection = bpy.data.collections.new('Robot')
        self.scene.collection.children.link(robot_collection)
        self.scene_map = {}
        
        ### ADD a light
        light_data = bpy.data.lights.new(name="light-data", type='AREA')
        light_data.energy = 100
        light_data.size = 3.5
        light_object = bpy.data.objects.new(name="light-object", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = (0, 0, 1.5)
        ###
        
        fk = self.robot.visual_trimesh_fk()
        for i, (tm, pose) in enumerate(fk.items()):
            mesh = tm if render_mesh else tm.bounding_box
            name = f'robot_mesh_{i}'
            bpy_mesh = bpy.data.meshes.new(name)
            bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=True)
            bpy_obj = bpy.data.objects.new(name, bpy_mesh)
            bpy_obj.rotation_mode = 'QUATERNION'
            bpy_obj.location = pose[:3,3]
            bpy_obj.rotation_quaternion = trimesh.transformations.quaternion_from_matrix(pose)
            bpy_obj.data.materials.append(mat)
            self.scene_map[tm] = bpy_obj
            robot_collection.objects.link(bpy_obj)

        ### ADD goal
        if self.viz_goal:
            # goal_mat = pyrender.MetallicRoughnessMaterial(
            #     metallicFactor=0.2,
            #     alphaMode='BLEND',
            #     baseColorFactor=(0.0, 1.0, 0.0, 0.3),
            # )
            goal_mat = bpy.data.materials.new(name="GoalMaterial")
            ### Pre-set Color
            # goal_mat.shadow_method = 'NONE'
            goal_mat.show_transparent_back = False
            hex_value = 0x58AF4A
            b = (hex_value & 0xFF) / 255.0
            g = ((hex_value >> 8) & 0xFF) / 255.0
            r = ((hex_value >> 16) & 0xFF) / 255.0
            goal_mat.use_nodes = True
            goal_mat.blend_method = 'BLEND'
            goal_principled = goal_mat.node_tree.nodes["Principled BSDF"]
            goal_principled.inputs["Base Color"].default_value = (r, g, b, 0.5)
            goal_principled.inputs["Alpha"].default_value = 0.5
            ###
            goal_collection = bpy.data.collections.new('Goal')
            self.scene.collection.children.link(goal_collection)
            fk = self.robot.visual_trimesh_fk(self.qgoal)
            for i, (tm, pose) in enumerate(fk.items()):
                mesh = tm if self.goal_use_mesh else tm.bounding_box
                name = f'robot_goal_{i}'
                bpy_mesh = bpy.data.meshes.new(name)
                bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=True)
                bpy_obj = bpy.data.objects.new(name, bpy_mesh)
                bpy_obj.rotation_mode = 'QUATERNION'
                bpy_obj.location = pose[:3,3]
                bpy_obj.rotation_quaternion = trimesh.transformations.quaternion_from_matrix(pose)
                bpy_obj.keyframe_insert(data_path="location", frame=0)
                bpy_obj.keyframe_insert(data_path="rotation_quaternion", frame=0)
                bpy_obj.data.materials.append(goal_mat)
                goal_collection.objects.link(bpy_obj)
                
            
            ### ADD initial configuration
            initial_mat = bpy.data.materials.new(name="InitialMaterial")
            ### Pre-set Color
            # goal_mat.shadow_method = 'NONE'
            initial_mat.show_transparent_back = False
            hex_value = 0xF5A65B
            b = (hex_value & 0xFF) / 255.0
            g = ((hex_value >> 8) & 0xFF) / 255.0
            r = ((hex_value >> 16) & 0xFF) / 255.0
            initial_mat.use_nodes = True
            initial_mat.blend_method = 'BLEND'
            initial_principled = initial_mat.node_tree.nodes["Principled BSDF"]
            initial_principled.inputs["Base Color"].default_value = (r, g, b, 0.5)
            initial_principled.inputs["Alpha"].default_value = 0.5
            ###
            initial_collection = bpy.data.collections.new('Initial')
            self.scene.collection.children.link(initial_collection)
            fk = self.robot.visual_trimesh_fk(self.init_qpos)
            for i, (tm, pose) in enumerate(fk.items()):
                mesh = tm if self.goal_use_mesh else tm.bounding_box
                name = f'robot_goal_{i}'
                bpy_mesh = bpy.data.meshes.new(name)
                bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=True)
                bpy_obj = bpy.data.objects.new(name, bpy_mesh)
                bpy_obj.rotation_mode = 'QUATERNION'
                bpy_obj.location = pose[:3,3]
                bpy_obj.rotation_quaternion = trimesh.transformations.quaternion_from_matrix(pose)
                bpy_obj.keyframe_insert(data_path="location", frame=0)
                bpy_obj.keyframe_insert(data_path="rotation_quaternion", frame=0)
                bpy_obj.data.materials.append(initial_mat)
                initial_collection.objects.link(bpy_obj)

    def _create_pyrender_scene(self, render_mesh, render_size, render_fps, renderer_kwargs, goal_mat=None, goal_wireframe=True):
        '''Create a pyrender scene for rendering.

        If using offscreen, creates cameras and lights similar to the pyrender interactive viewer.

        Args:
            render_mesh: bool
                Whether to render the mesh.
            render_size: Tuple[int, int]
                The size of the render.
            render_fps: int
                The frames per second to render.
            renderer_kwargs: dict
                Extra kwargs to pass to the renderer.
        '''
        import pyrender
        self.scene = pyrender.Scene()
        fk = self.robot.visual_trimesh_fk()
        self.scene_map = {}
        for tm, pose in fk.items():
            if render_mesh:
                mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            else:
                mesh = pyrender.Mesh.from_trimesh(tm.bounding_box, smooth=False)
            node = self.scene.add(mesh, pose=pose)
            self.scene_map[tm] = node
        ### ADD the goal
        if self.viz_goal:
            if goal_mat is None:
                goal_mat = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='BLEND',
                    baseColorFactor=(0.0, 1.0, 0.0, 0.3),
                )
            fk = self.robot.visual_trimesh_fk(self.qgoal)
            for tm, pose in fk.items():
                if self.goal_use_mesh:
                    if goal_wireframe:
                        mesh = pyrender.Mesh.from_trimesh(tm, smooth=False, wireframe=True, material=goal_mat)
                        self.scene.add(mesh, pose=pose)
                    mesh = pyrender.Mesh.from_trimesh(tm, smooth=False, material=goal_mat)
                    self.scene.add(mesh, pose=pose)
                else:
                    if goal_wireframe:
                        mesh = pyrender.Mesh.from_trimesh(tm.bounding_box, smooth=False, wireframe=True, material=goal_mat)
                        self.scene.add(mesh, pose=pose)
                    mesh = pyrender.Mesh.from_trimesh(tm.bounding_box, smooth=False, material=goal_mat)
                    self.scene.add(mesh, pose=pose)
                    
        if self.renderer == RENDERER.PYRENDER:
            # Default viewport arguments
            kwargs = {
                'viewport_size': render_size,
                'run_in_thread': True,
                'use_raymond_lighting': True,
                'refresh_rate': render_fps
            }
            kwargs.update(renderer_kwargs)
            self.scene_viewer = pyrender.Viewer(
                                    self.scene,
                                    **kwargs)
            self.scene_viewer.render_lock.acquire()
        else:
            w, h = render_size
            ### compute a camera (from the pyrender viewer code)
            from pyrender.constants import DEFAULT_Z_FAR, DEFAULT_Z_NEAR, DEFAULT_SCENE_SCALE
            ## camera pose
            centroid = self.scene.centroid
            if renderer_kwargs.get('view_center') is not None:
                centroid = renderer_kwargs['view_center']
            scale = self.scene.scale    
            if scale == 0.0:
                scale = DEFAULT_SCENE_SCALE
            s2 = 1.0 / np.sqrt(2.0)
            cp = np.eye(4)
            cp[:3,:3] = np.array([
                [0.0, -s2, s2],
                [1.0, 0.0, 0.0],
                [0.0, s2, s2]
            ])
            hfov = np.pi / 6.0
            dist = scale / (2.0 * np.tan(hfov))
            cp[:3,3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
            ## camera perspective
            zfar = max(self.scene.scale * 10.0, DEFAULT_Z_FAR)
            if self.scene.scale == 0:
                znear = DEFAULT_Z_NEAR
            else:
                znear = min(self.scene.scale / 10.0, DEFAULT_Z_NEAR)
            ## camera node
            camera = pyrender.camera.PerspectiveCamera(yfov=np.pi / 3.0, znear=znear, zfar=zfar)
            camera_node = pyrender.Node(matrix=cp, camera=camera)
            self.scene.add_node(camera_node)
            self.scene.main_camera_node = camera_node
            ## raymond lights
            thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
            phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
            for phi, theta in zip(phis, thetas):
                xp = np.sin(theta) * np.cos(phi)
                yp = np.sin(theta) * np.sin(phi)
                zp = np.cos(theta)

                z = np.array([xp, yp, zp])
                z = z / np.linalg.norm(z)
                x = np.array([-z[1], z[0], 0.0])
                if np.linalg.norm(x) == 0:
                    x = np.array([1.0, 0.0, 0.0])
                x = x / np.linalg.norm(x)
                y = np.cross(z, x)

                matrix = np.eye(4)
                matrix[:3,:3] = np.c_[x,y,z]
                self.scene.add_node(pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix
                ))
            ### Camera computed

            kwargs = OrderedDict()
            kwargs['viewport_width'] = w
            kwargs['viewport_height'] = h
            kwargs.update(renderer_kwargs)
            self.scene_viewer = pyrender.OffscreenRenderer(**kwargs)

    @property
    def action_spec(self):
        '''Gym Compat'''
        pass
    @property
    def action_dim(self):
        '''Gym Compat'''
        pass
    @property 
    def action_space(self):
        '''Gym Compat'''
        pass 
    @property 
    def observation_space(self):
        '''Gym Compat'''
        pass 
    @property 
    def obs_dim(self):
        '''Gym Compat'''
        pass

    @property
    def info(self):
        '''Get a deepcopy of the info of the environment.'''
        return copy.deepcopy(self._info)

    # State related properties and setters
    @property
    def qpos(self) -> np.ndarray:
        '''Get the joint positions of the URDF robot.'''
        return self.state[:self.dof]
    
    @qpos.setter
    def qpos(self, val: np.ndarray):
        '''Set the joint positions of the URDF robot.'''
        self.state[:self.dof] = val

    @property
    def qvel(self) -> np.ndarray:
        '''Get the joint velocities of the URDF robot.'''
        return self.state[self.dof:]
    
    @qvel.setter
    def qvel(self, val: np.ndarray):
        '''Set the joint velocities of the URDF robot.'''
        self.state[self.dof:] = val
    
    @property
    def init_qpos(self) -> np.ndarray:
        '''Get the initial positions'''
        return self.init_state[:self.dof]

    @property
    def init_qvel(self) -> np.ndarray:
        '''Get the initial velocities'''
        return self.init_state[self.dof:]
    
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym import logger
from typing import Optional
class FullStepRecorder(VideoRecorder):
    """Wrapper on the gym video recorder which saves all frames from a given step and supports selecting the camera id as a tuple of the return value from the render function.
    """
    def __init__(
        self,
        env,
        camera_id: int = 0,
        path: Optional[str] = None,
        metadata: Optional[dict] = None,
        enabled: bool = True,
        base_path: Optional[str] = None,
        **kwargs
    ):
        """Video recorder renders a nice movie of a rollout, frame by frame.

        Args:
            env (Env): Environment to take video of.
            camera_id (int): Camera ID to use for recording if a tuple of camera frames is returned. 0 indicates the first camera or tuple entry.
            path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
            metadata (Optional[dict]): Contents to save to the metadata file.
            enabled (bool): Whether to actually record video, or just no-op (for convenience)
            base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.

        Raises:
            Error: You can pass at most one of `path` or `base_path`
            Error: Invalid path given that must have a particular file extension
        """
        temp = getattr(env, "render_mode", "rgb_array")
        env.render_mode = "rgb_array"
        super().__init__(env, path=path, metadata=metadata, enabled=enabled, base_path=base_path, **kwargs)
        env.render_mode = temp
        self.camera_id = camera_id

    def capture_frame(self, **kwargs):

        frames = self.env.render(**kwargs)
        if isinstance(frames, tuple):
            frames = frames[self.camera_id]

        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        if frames is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn(
                    "Env returned None on `render()`. Disabling further rendering for video recorder by marking as "
                    f"disabled: path={self.path} metadata_path={self.metadata_path}"
                )
                self.broken = True
        elif isinstance(frames, list):
            self.recorded_frames.extend(frames)
        else:
            self.recorded_frames.append(frames)
            
def get_next_available_filename(base_path):
    directory, filename = os.path.split(base_path)
    name, extension = os.path.splitext(filename)
    
    counter = 0
    new_path = os.path.join(directory, f"{name}{extension}")

    while os.path.exists(new_path) and os.path.isfile(new_path):
        counter += 1
        new_path = os.path.join(directory, f"{name}_{counter}{extension}")

    return new_path

class RobotStepViz:
    def __init__(self, save_every_n_iteration=5):
        self.mat = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=(1.0, 1.0, 1.0, 0.3),
        )
        self.mesh_list = []
        self.save_every_n_iteration = save_every_n_iteration
    
    def render_callback(self, env):
        if len(self.mesh_list) == 0:
            # add the initial mesh
            init_mesh_list = self.get_tm_list(env, env.init_qpos)
            self.mesh_list.append(init_mesh_list)
        
        if env.renderer == RENDERER.PYRENDER or env.renderer == RENDERER.PYRENDER_OFFSCREEN:
            # Create the nodes
            pyrender_mesh_list = []
            total_meshes = len(self.mesh_list)
            for i, mesh in enumerate(self.mesh_list):
                mat = self.mat
                mat.baseColorFactor = (mat.baseColorFactor[0], mat.baseColorFactor[1], mat.baseColorFactor[2], 1.0 - 0.6*(total_meshes-i)/total_meshes)
                pyr_mesh = pyrender.Mesh.from_trimesh(mesh, material=mat)
                pyrender_mesh_list.append(pyr_mesh)
            node_list = [env.scene.add(mesh) for mesh in pyrender_mesh_list]

            # Add the current mesh to the list
            current_mesh_list = self.get_tm_list(env, env.qpos)
            self.mesh_list.append(current_mesh_list)
            return lambda: [env.scene.remove_node(node) for node in node_list]
        
        elif env.renderer == RENDERER.BLENDER:
            import bpy
            self.itr = getattr(self, 'itr', 1)
            if self.itr % self.save_every_n_iteration == 1:
                # Add the current mesh to the list
                current_mesh_list = self.get_tm_list(env, env.qpos)
                self.mesh_list.append(current_mesh_list)
                
                traj_mat = bpy.data.materials.get("RobotTrajMat")
                if traj_mat is None:
                    traj_mat = bpy.data.materials.new(name="RobotTrajMat")
                    traj_mat.use_nodes = True
                    traj_mat.blend_method = 'OPAQUE'
                    traj_mat.show_transparent_back = False
                    traj_mat.use_nodes = True
                    robot_principled = traj_mat.node_tree.nodes["Principled BSDF"]
                    robot_principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 0.3)
                    robot_principled.inputs["Alpha"].default_value = 0.3 #
                        
                itr = getattr(self, 'itr', 1)
                viz_collection = getattr(self, 'viz_collection', None)
                if viz_collection is None:
                    viz_collection = bpy.data.collections.new('robot_traj_viz')
                    env.scene.collection.children.link(viz_collection)
                    self.viz_collection = viz_collection
                
                viz_subcollection = bpy.data.collections.new(f'robot_traj_viz_{itr}')
                viz_collection.children.link(viz_subcollection)
                current_frame = env.scene.frame_current
                self.all_last_obj = []
                
                for i, mesh in enumerate(current_mesh_list):
                    name = f'robot_traj_mesh_step{itr}_{i}'
                    bpy_mesh = bpy.data.meshes.new(name)
                    bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=False)
                    bpy_mesh.validate()
                    bpy_mesh.update()
                    bpy_obj = bpy.data.objects.new(name, bpy_mesh)
                    # bpy_obj.hide_render = True
                    # bpy_obj.hide_viewport = True
                    
                    ## original
                    bpy_obj.keyframe_insert(data_path="hide_render", frame=0)
                    bpy_obj.keyframe_insert(data_path="hide_viewport", frame=0)
                    bpy_obj.data.materials.append(traj_mat)
                    viz_subcollection.objects.link(bpy_obj)
                    self.all_last_obj.append(bpy_obj)
                
            self.itr += 1

            def hide_all():
                return
                for obj in self.all_last_obj:
                    obj.hide_render = True
                    obj.hide_viewport = True
                    obj.keyframe_insert(data_path="hide_render")
                    obj.keyframe_insert(data_path="hide_viewport")

            return hide_all
            
    def get_tm_list(self, env, qpos):
        fk = env.robot.visual_trimesh_fk(qpos)
        tm_list = []
        for tm, pose in fk.items():
            mesh = tm.copy()
            mesh.apply_transform(pose)
            tm_list.append(mesh)
        return tm_list
    
if __name__ == '__main__':
    robots2.DEBUG_VIZ = False
    robot = robots2.ZonoArmRobot.load(os.path.join(os.getcwd(),'envs/arm_urdfs/ur5_kinova/ur5_kinova_gen3.urdf'), create_joint_occupancy=False)
    timestep_discretization = 10
    env = DualArmEnv(robot=robot.urdf,
                            timestep_discretization=timestep_discretization,
                            step_type='direct',
                            check_self_collision=False,
                            verbose_self_collision=True,
                            seed=0)
    env.reset()
    step_time = 0.1
    env.render()
    for i in range(100):
        qpos = env.qpos
        velocity_action = (np.random.random(12) - 0.5) * 0.
        velocity_action = np.tile(velocity_action, timestep_discretization).reshape(timestep_discretization, -1)
        configuration = qpos + np.linspace(0., step_time, num=timestep_discretization ,endpoint=True).reshape(-1,1) * velocity_action
        env.step((configuration, velocity_action))
        env.render()
    env.close()