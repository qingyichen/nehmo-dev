import torch
import numpy as np
from torch.autograd import grad

import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import cyipopt


def wrap_joint(configs):
    return (configs + torch.pi) % (2 * torch.pi) - torch.pi

class FiveArmAgent():
    def __init__(self, 
                 max_joint_velocity=0.5,
                 num_links=6,
                 val_func_model=None,
                 verbose=False,
                 device='cpu') -> None:
        self.device = device
        self.max_joint_velocity = max_joint_velocity
        self.model = val_func_model
        self.state_dim = num_links
        self.verbose = verbose

    def plan(self, agent_state, other_agent_state, goal_state, step_time=0.1, safe_time=0.3, buffer=0.0, planner_mode='hji'):
        if not isinstance(agent_state, torch.Tensor):
            agent_state = torch.tensor(agent_state, device=self.device)
        if not isinstance(other_agent_state, torch.Tensor):
            other_agent_state = torch.tensor(other_agent_state, device=self.device)
        if not isinstance(goal_state, torch.Tensor):
            goal_state = torch.tensor(goal_state, device=self.device)
        num_constraints = other_agent_state.shape[0] if other_agent_state.ndim > 1 else 1

        if planner_mode == 'simple':
            return self.make_simple_plans(current_state=agent_state, goal_state=goal_state)
            
        # update adversary positions such that each agent will act as if they are adversary
        adversary_positions = other_agent_state
        adversary_predicted_actions = self.predict_adversary_plans(self_state=agent_state, adversary_agent_state=other_agent_state, time=step_time)
        adversary_positions += adversary_predicted_actions.flatten() * step_time if adversary_positions.ndim == 1 else adversary_predicted_actions * step_time
        adversary_positions = wrap_joint(adversary_positions)
        nlp_obj = NNArm_NLP(
            num_links=self.state_dim,
            start_position=agent_state,
            goal_position=goal_state,
            max_joint_velocity=self.max_joint_velocity,
            step_time=step_time,
            safe_time=safe_time,
            val_func_model=self.model,
            device=self.device,
            adversary_positions=adversary_positions,
        )

        nlp = cyipopt.Problem(
            n=self.state_dim,
            m=num_constraints,
            problem_obj=nlp_obj,
            lb=[-self.max_joint_velocity]*self.state_dim,
            ub=[self.max_joint_velocity]*self.state_dim,
            cl=[buffer]*num_constraints,
            cu=[1e20]*num_constraints, 
        )
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)
        nlp.add_option('max_iter', 30)
        try:
            # if optimization fails (when problem infeasible or problem exceeds time limit)
            # actiavte the action such the HJI value function is maximized
            k_opt, self.info = nlp.solve(np.zeros(self.state_dim))
        except:
            return self.make_safe_plans().flatten()

        if (self.info['status'] != 0): 
            if self.verbose:
                print(f"Failed to find a solution... using safe plan.")
            return self.make_safe_plans().flatten()
        return torch.tensor(k_opt)
    
    def predict_adversary_plans(self, self_state, adversary_agent_state, time=0.1):
        coords = torch.zeros(adversary_agent_state.shape[0] if adversary_agent_state.ndim > 1 else 1, self.state_dim * 2 + 1, device=self.device)
        coords[:,0] = time
        coords[:,1:1+self.state_dim] = self_state
        # for triple arm
        if coords.shape[0] > 1:
            coords[1:2,1] = wrap_joint(coords[1:2,1] + torch.pi)
            coords[2:3,1] = wrap_joint(coords[2:3,1] - torch.pi / 2)
            coords[3:4,1] = wrap_joint(coords[3:4,1] + torch.pi / 2)
        coords[:,1+self.state_dim:] = adversary_agent_state
        self.currenrt_state_values, self.current_state_grads = self.predict_value_function(coords, compute_grad=True)
        adversary_plans = -torch.sign(self.current_state_grads[:,1+self.state_dim:]) * self.max_joint_velocity
        
        return adversary_plans
    
    def make_simple_plans(self, current_state, goal_state, step_time=0.1):
        # simply pick the action that brings closest to the goal
        diff_state = (goal_state - current_state)
        continuous_joint_flag = np.array([True, True, True, True, True, True]) * False
        diff_state[continuous_joint_flag] = wrap_joint(diff_state[continuous_joint_flag])
        simple_plan = torch.clamp(diff_state / step_time, min=-self.max_joint_velocity, max=self.max_joint_velocity)
        return simple_plan
    
    def make_safe_plans(self):
        # pick the safeset action according to the value function
        safe_plan = torch.sign(self.current_state_grads[:,1:1+self.state_dim]) * self.max_joint_velocity
        safe_plan = safe_plan[torch.argmin(self.currenrt_state_values)]
        return safe_plan
    
    def predict_value_function(self, coords=None, compute_grad=False):
        model_outputs = self.model({'coords': coords})
        gradient = None
        values, inputs = model_outputs['model_out'], model_outputs['model_in']
        if compute_grad:
            gradient = grad(values, inputs, grad_outputs=torch.ones_like(values))[0]
            symmetry_mask = model_outputs['symmetry_mask']
            if symmetry_mask is not None:
                gradient[symmetry_mask] *= -1
        return values, gradient
    
    
class NNArm_NLP:
    def __init__(self, 
                 start_position=None,
                 goal_position=None,
                 num_links=6,
                 max_joint_velocity=0.5,
                 step_time=0.1,
                 safe_time=0.5,
                 val_func_model=None,
                 device='cpu',
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
        else:
            self.adversary_states = torch.tensor(adversary_positions).to(device)
        
        self.num_links = num_links
        self.np_state = self.state.cpu().numpy()
        self.np_adversary_states = self.adversary_states.cpu().numpy()
        self.np_goal = self.goal_state.cpu().numpy()

        self.num_var = num_links
        self.max_joint_velocity = max_joint_velocity
        self.step_time = step_time
        self.safe_time = safe_time
        self.model = val_func_model
        self.device = device
        self.prev_x = np.zeros_like(self.np_state) * np.nan
        
        self.continuous_joint_flag = np.array([True, True, True, True, True, True]) * False
        
        self.num_constraints = 4
        return 
    
    def objective(self, x):
        new_state = self.np_state + self.step_time * x
        diff_state = (new_state - self.np_goal)
        diff_state[self.continuous_joint_flag] = wrap_joint(diff_state[self.continuous_joint_flag])
        return np.sum(np.square(diff_state))

    def gradient(self, x):
        new_state = self.np_state + self.step_time * x
        diff_state = (new_state - self.np_goal)
        diff_state[self.continuous_joint_flag] = wrap_joint(diff_state[self.continuous_joint_flag])
        return 2 * self.step_time * diff_state

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
            coords = torch.zeros(self.adversary_states.shape[0] if self.adversary_states.ndim > 1 else 1, self.num_links * 2 + 1, device=self.device)
            coords[:,0] = self.safe_time
            coords[:,1:1+self.num_links] = self.state
            if coords.shape[0] > 1:
                coords[1:2,1] = wrap_joint(coords[1:2,1] + torch.pi)
                coords[2:3,1] = wrap_joint(coords[2:3,1] - torch.pi / 2)
                coords[3:4,1] = wrap_joint(coords[3:4,1] + torch.pi / 2)
            coords[:,1:1+self.num_links] += torch.tensor(x, device=self.device) * self.step_time
            coords[:,1+self.num_links:] = self.adversary_states
            model_outputs = self.model({'coords': coords})
            values, inputs = model_outputs['model_out'], model_outputs['model_in']
            gradient = grad(values, inputs, grad_outputs=torch.ones_like(values))[0][:,1:1+self.num_links] * self.step_time
            symmetry_mask = model_outputs['symmetry_mask']
            if symmetry_mask is not None:
                gradient[symmetry_mask] *= -1

            self.cons = values.detach().cpu().numpy().flatten()
            self.jac = gradient.detach().cpu().numpy()
            self.prev_x = x.copy()
        return