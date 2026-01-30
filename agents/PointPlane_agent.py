import torch
import numpy as np
from torch.autograd import grad

import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from deepreach import modules
import cyipopt

class PointPlaneAgent():
    def __init__(self, 
                 velocity_limit=0.5,
                 step_time=0.1,
                 safe_time=0.3,
                 val_func_model=None,
                 device='cpu') -> None:
        self.safe_time = safe_time
        self.step_time = step_time
        self.device = device
        self.velocity_limit = velocity_limit
        self.model = val_func_model
        self.state_dim = 2

    def plan(self, agent_state, goal_state, other_agent_states=None, static_obs_positions=None, static_obs_sizes=None, buffer=0., planner_mode='hji'):
        num_static_obstacles = len(static_obs_positions) if static_obs_positions is not None else 0
        if planner_mode == 'hji':
            num_other_agents = other_agent_states.shape[0]
            num_constarints = num_static_obstacles + num_other_agents

            # update adversary positions such that each agent will act as if they are adversary
            adversary_positions = other_agent_states.to(self.device)
            adversary_predicted_actions = self.predict_adversary_plans(self_state=agent_state, adversary_states=other_agent_states)
            adversary_positions += adversary_predicted_actions * self.step_time
        elif planner_mode == 'simple':
            num_constarints = num_static_obstacles
            num_other_agents = 0
        nlp_obj = pointPlane_NLP(
            start_position=agent_state,
            goal_position=goal_state,
            velocity_limit=self.velocity_limit,
            step_time=self.step_time,
            safe_time=self.safe_time,
            val_func_model=self.model,
            device=self.device,
            num_other_agents=num_other_agents,
            num_static_obstacles=num_static_obstacles,
            adversary_positions=adversary_positions if planner_mode == 'hji' else None,
        )

        nlp = cyipopt.Problem(
            n=self.state_dim,
            m=num_constarints,
            problem_obj=nlp_obj,
            lb=[-self.velocity_limit]*self.state_dim,
            ub=[self.velocity_limit]*self.state_dim,
            cl=[buffer]*num_constarints,
            cu=[1e20]*num_constarints, 
        )
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)
        nlp.add_option('max_iter', 30)
        k_opt, self.info = nlp.solve(np.zeros(self.state_dim))

        # if optimization fails (when problem infeasible or problem exceeds time limit)
        # actiavte the safe action of stopping
        if planner_mode == 'simple':
            return torch.tensor(k_opt, device=self.device)
        if self.info['status'] != 0:
            min_val_index = torch.argmin(self.currenrt_state_values)
            min_val_grads = self.current_state_grads[min_val_index]
            k_opt = torch.sign(min_val_grads[1:3]) * self.velocity_limit * 0.
            return k_opt
        return torch.tensor(k_opt, device=self.device)
    
    def predict_adversary_plans(self, self_state, adversary_states):
        num_adversary_agents = adversary_states.shape[0]
        coords = torch.zeros(num_adversary_agents, 5, device=self.device)
        coords[:,0] = self.safe_time
        coords[:,1:3] = self_state
        coords[:,3:5] = adversary_states
        self.currenrt_state_values, self.current_state_grads = self.predict_value_function(coords, compute_grad=True)
        adversary_plans = -torch.sign(self.current_state_grads[:,3:5]) * self.velocity_limit
        
        return adversary_plans
    
    def predict_value_function(self, coords=None, compute_grad=False):
        model_outputs = self.model({'coords': coords})
        gradient = None
        values, inputs = model_outputs['model_out'], model_outputs['model_in']
        if compute_grad:
            gradient = grad(values, inputs, grad_outputs=torch.ones_like(values))[0]
        return values, gradient
    
        
class pointPlane_NLP:
    def __init__(self, 
                 start_position=None,
                 goal_position=None,
                 velocity_limit=0.5,
                 step_time=0.1,
                 safe_time=0.5,
                 val_func_model=None,
                 device='cpu',
                 num_other_agents=0,
                 num_static_obstacles=0,
                 adversary_positions=None
                 ):
        if isinstance(start_position, torch.Tensor):
            self.state = start_position.to(device)
        else:
            self.state = torch.tensor(start_position, device=device)
        if isinstance(goal_position, torch.Tensor):
            self.goal_state = goal_position.to(device)
        else:
            self.goal_state = torch.tensor(goal_position, device=device)
        if isinstance(adversary_positions, torch.Tensor):
            self.adversary_states = adversary_positions.to(device)
        elif adversary_positions is not None:
            self.adversary_states = torch.cat(adversary_positions, dim=0)
        else:
            self.adversary_states = None
        
        self.np_state = self.state.cpu().numpy()
        self.np_adversary_states = self.adversary_states.cpu().numpy() if self.adversary_states is not None else None
        self.np_goal = self.goal_state.cpu().numpy()

        self.num_var = self.state.shape[0]
        self.velocity_limit = velocity_limit
        self.step_time = step_time
        self.safe_time = safe_time
        self.model = val_func_model
        self.device = device
        self.prev_x = np.zeros_like(self.np_state) * np.nan
        
        self.num_constraints = num_static_obstacles + num_other_agents
        self.num_other_agents = num_other_agents
        return 
    
    def objective(self, x):
        new_state = self.np_state + self.step_time * x
        return np.sum(np.square(new_state - self.np_goal))

    def gradient(self, x):
        new_state = self.np_state + self.step_time * x
        return 2 * self.step_time * (new_state - self.np_goal)

    def constraints(self, x):
        self.compute_constraints_jacobian(x)
        return self.cons

    def jacobian(self, x):
        self.compute_constraints_jacobian(x)
        return self.jac
    
    def compute_constraints_jacobian(self, x):
        if (self.prev_x != x).any():
            self.cons = torch.zeros((self.num_constraints))
            self.jac = torch.zeros((self.num_constraints, self.num_var))

            # constraint with other agents
            coords = torch.zeros(self.num_other_agents, 5, device=self.device)
            coords[:,0] = self.safe_time
            coords[:,1:3] = self.state + torch.tensor(x, device=self.device) * self.step_time
            coords[:,3:5] = self.adversary_states
            model_outputs = self.model({'coords': coords})
            values, inputs = model_outputs['model_out'], model_outputs['model_in']
            gradient = grad(values, inputs, grad_outputs=torch.ones_like(values))[0][:,1:3] * self.step_time

            self.cons = values.detach().cpu().numpy().flatten()
            self.jac = gradient.detach().cpu().numpy()
            self.prev_x = x.copy()
        return
        