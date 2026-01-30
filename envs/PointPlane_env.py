import os
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

class PointPlaneEnv():
    def __init__(
        self,
        robot_size=0.1,
        dimension=2,
        num_random_agents=2,
        num_static_obstacles=0,
        obs_size_max=0.5,
        obs_size_min=0.05,
        velocity_limit=0.5,
        step_time=0.1, 
        safe_time=0.5,
        collision_threshold=1e-6,
        goal_threshold=0.01,
        dtype=torch.float,
        device='cpu',
        seed=0,
    ) -> None:
        super().__init__()
        
        # robot
        self.robot_size = robot_size
        self.velocity_limit = velocity_limit
        self.num_agents = num_random_agents

        # obstacles
        self.num_static_obstacles = num_static_obstacles
        self.obs_size_max = obs_size_max
        self.obs_size_min = obs_size_min

        # collision
        self.collision_threshold = collision_threshold

        # env
        self.dimension = dimension
        self.bound = self.num_agents * 0.25
        self.goal_threshold = goal_threshold
        self.dtype = dtype
        self.step_time = step_time
        self.safe_time = safe_time
        self.device = device
        
        # rendering
        self.first_time_rendering = True
        self.frame_step = 0
        self.save_dir = f'planning_videos/PointPlane/{self.num_agents}agents'
        self.rendered_frames = []

        self.reset(seed=seed)


    def generate_random_static_obstacles(self, num_static_obstacles=1, obs_size_max=0.5, obs_size_min=0.05):
        # obsatcle positions
        self.obs_pos = (
            (torch.rand(num_static_obstacles, self.dimension, device=self.device) - 0.5)
            * 2
            * self.bound
        )
        self.obs_size = (
            torch.rand(num_static_obstacles, device=self.device)
            * (obs_size_max - obs_size_min)
            + self.obs_size_min
        )
        return
    
    def generate_random_agents(self, num_random_agents=2, distance_threshold=0.05):
        start_state_settled = False
        while not start_state_settled:
            self.agent_start_states = (torch.rand(num_random_agents, self.dimension) - 0.5) * 2 * (self.bound - 0.1)
            pairwise_distances = torch.cdist(self.agent_start_states, self.agent_start_states)
            pairwise_distances[range(self.num_agents), range(self.num_agents),]= torch.inf
            start_state_settled = torch.all(pairwise_distances > distance_threshold + 2 * self.robot_size)
        self.agent_states = self.agent_start_states.clone()

        goal_state_settled = False
        while not goal_state_settled:
            self.agent_goal_states = (torch.rand(num_random_agents, self.dimension) - 0.5) * 2 * (self.bound - 0.1)
            pairwise_distances = torch.cdist(self.agent_goal_states, self.agent_goal_states)
            pairwise_distances[range(self.num_agents), range(self.num_agents),]= torch.inf
            goal_state_settled = torch.all(pairwise_distances > distance_threshold + 2 * self.robot_size)
        self.goal_states = self.agent_goal_states.clone()
            
        return

    def reset(self, seed=0):
        torch.manual_seed(seed=seed)
        self.generate_random_agents(num_random_agents=self.num_agents)
        self.generate_random_static_obstacles(num_static_obstacles=self.num_static_obstacles, 
                                              obs_size_max=self.obs_size_max, 
                                              obs_size_min=self.obs_size_min)
        self.trajectory_history = [self.agent_states.tolist()]
        return

    
    def reset_initial_condition(self):
        return

    def step(self, action=None):
        self.agent_states += action * self.step_time
        observation = self.agent_states.clone()
        reward = 0.
        done = False
        success = self.check_success()
        collision = self.check_collision()
        self.trajectory_history.append(self.agent_states.tolist())
        
        return observation, reward, done, {'success': success, 'collision': collision}

    def render(self, close=False, save=False, save_name='', show=False, refined=False):
        if self.first_time_rendering:
            self.first_time_rendering = False
            if not refined:
                self.fig = plt.figure()
            else:
                self.fig = plt.figure(dpi=600)
            self.ax = self.fig.add_subplot()
            self.render_frame(ax=self.ax)
            self.fig.canvas.draw()
        else:
            self.ax.cla()
            self.render_frame(ax=self.ax)
            self.fig.canvas.draw()
            if show:
                plt.pause(0.01)
        if close:
            if save and len(self.rendered_frames) > 0:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                imageio.mimsave(os.path.join(self.save_dir, f"{save_name}.gif"), self.rendered_frames)
            plt.close()
            save_final_frame = refined ##
            if save and save_final_frame:
                imageio.imwrite(os.path.join(self.save_dir, f"{save_name}_start.png"), self.rendered_frames[0])
                imageio.imwrite(os.path.join(self.save_dir, f"{save_name}_end.png"), self.rendered_frames[-1])
        if save:
            image = np.array(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            self.rendered_frames.append(image)
    
    def render_frame(self, positions=None, ax=None, visualize_goal=True):
        if positions is None:
            positions = self.agent_states.cpu().numpy()
        
        cs = np.linspace(-0.15, 0.1, self.num_agents)
        cs[cs <= -0.1] += 0.83
        cs[cs <= 0.] += 1.
        colors = cm.rainbow(cs)
        
        visualize_history = False ##
        spacing = 6
        if visualize_history:
            trajectory_history = np.array(self.trajectory_history)
            for i in range(self.num_agents):
                trajectory = trajectory_history[::spacing,i,:]
                ax.scatter(trajectory[:,0], trajectory[:,1], color=colors[i], s=40 // np.sqrt(self.num_agents), alpha=np.linspace(0., 1., num=trajectory.shape[0]))
        
        # Plot every robot's current positions
        circles = [plt.Circle((pos[0], pos[1]), self.robot_size, fill = True, color=colors[i], linewidth=0.5, alpha = 0.7) for i, pos in enumerate(positions)]
        # Plot every robot's goal
        positions = self.agent_goal_states.cpu().numpy()
        circles += [plt.Circle((pos[0], pos[1]), self.robot_size, fill = False, color=colors[i], linewidth=0.7, linestyle='--') for i, pos in enumerate(positions)]
        for c in circles:
            ax.add_patch(c)
            
        ax.set_aspect('equal')
        ax.set_xlim([-self.bound, self.bound])
        ax.set_ylim([-self.bound, self.bound])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(w_pad=0.5, h_pad=0.5)
        return

    def check_collision(self) -> bool:
        pairwise_distances = torch.cdist(self.agent_states, self.agent_states)
        pairwise_distances[range(self.num_agents), range(self.num_agents),]= torch.inf
        agent_collision = torch.any(pairwise_distances < self.robot_size * 2)
        return agent_collision.item()
    
    def check_success(self):
        agents_success = torch.linalg.norm(self.agent_states - self.agent_goal_states, dim=1) < self.goal_threshold
        return torch.all(agents_success).item()

if __name__ == "__main__":
    num_agents = 5
    env = PointPlaneEnv(num_random_agents=num_agents, device='cuda')
    env.reset(seed=0)
    
    num_steps = 200
    for i in range(num_steps):
        print(i)
        env.step(torch.rand(num_agents, 2) - 0.5)
        env.render(close=(i==num_steps-1), save=True, save_name='test', show=False)
