import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math
import pytorch_kinematics as pk
import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from deepreach.utils import line_segment_distances
from torch.autograd import grad



class BatchLinear(nn.Linear):
    '''A linear layer'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords)
        return output


class SingleBVPNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}
    
    
class PointPlane_ICNet(SingleBVPNet):
    def __init__(self, out_features=1, type='sine', in_features=4,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, max_velocty=0.5, collisionR=0.2,
                 provide_initial_condition=False, **kwargs):
        super().__init__(out_features, type, in_features, mode, hidden_features, num_hidden_layers, **kwargs)
        self.max_velocity = max_velocty
        self.provide_initial_condition = provide_initial_condition
        self.collisionR = collisionR
        
    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        if not self.provide_initial_condition:
            output = self.net(coords)
            return {'model_in': coords_org, 'model_out': output}
        else:
            boundary_values = torch.norm(coords[..., 1:3] - coords[..., 3:5], dim=-1, keepdim=True) - self.collisionR
            # no training is required for solving PointPlane problem as the boundary condition solves the HJ PDE
            return {'model_in': coords_org, 'model_out': boundary_values} # + 0. * self.net(coords)
    
    
class NN_SymmetryICNet(SingleBVPNet):
    def __init__(self, out_features=1, type='sine', in_features=12, mode='mlp', hidden_features=256, num_hidden_layers=3, 
                 bc_model=None, use_symmetry=False, num_links=12, max_joint_velocity=0.5,
                 provide_initial_condition=False,
                 **kwargs):
        super().__init__(out_features, type, in_features, mode, hidden_features, num_hidden_layers, **kwargs)
        
        self.num_links = num_links
        self.max_joint_velocity = max_joint_velocity
        
        self.use_symmetry = use_symmetry
        self.provide_initial_condition = provide_initial_condition
        
        self.bc_model = bc_model
        self.ReLu = nn.ReLU()
    
    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        
        symmetry_mask = None
        if self.use_symmetry:
            # symmetry: V(q11, q12, ..., q21, q22, ...) = V(pi-q11, -q12, ..., pi-q21, -q22, ...)
            states = model_input['coords'][...,1:]
            symmetry_mask = torch.logical_or(states[..., 6] < -torch.pi / 2, states[..., 6] > torch.pi / 2)                
            temp = model_input['coords'][symmetry_mask]
            temp *= -1
            temp[..., 1] += torch.pi
            temp[..., 7] += torch.pi
            temp[..., 1] = (temp[..., 1] + torch.pi) % (2 * torch.pi) - torch.pi
            temp[..., 7] = (temp[..., 7] + torch.pi) % (2 * torch.pi) - torch.pi
            model_input['coords'][symmetry_mask] = temp
        
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        
        if self.provide_initial_condition:
            states = coords_org[...,1:]
            boundary_condition = self.bc_model(states)
            output = boundary_condition + self.net(coords)            
            return {'model_in': coords_org, 'model_out': output, 'symmetry_mask': symmetry_mask}
        else:
            output = self.net(coords)
            return {'model_in': coords_org, 'model_out': output, 'symmetry_mask': symmetry_mask}
    
    
class SimpleArm_SymmetryICNet(SingleBVPNet):
    def __init__(self, out_features=1, type='sine', in_features=2, mode='mlp', hidden_features=256, num_hidden_layers=3, 
                 robot1_urdf_path='', robot2_urdf_path='', collision_R=0.1, use_symmetry=False, num_links=2, max_joint_velocity=0.5,
                 provide_initial_condition=False,
                 **kwargs):
        super().__init__(out_features, type, in_features, mode, hidden_features, num_hidden_layers, **kwargs)
        
        self.num_links = num_links
        self.max_joint_velocity = max_joint_velocity
        self.collision_R = collision_R
        
        self.use_symmetry = use_symmetry
        self.provide_initial_condition = provide_initial_condition
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.chain1 = pk.build_serial_chain_from_urdf(open(robot1_urdf_path).read(), f"link{self.num_links+1}").to(device=device)
        self.chain2 = pk.build_serial_chain_from_urdf(open(robot2_urdf_path).read(), f"link{self.num_links+1}").to(device=device)
    
        
    def boundary_condition(self, states):
        q1 = states[..., :self.num_links]
        q2 = states[..., self.num_links:]
        original_shape = q1.shape[:-1]
                
        fk1 = self.chain1.forward_kinematics(q1.view(-1, self.num_links), end_only=False)
        fk2 = self.chain2.forward_kinematics(q2.view(-1, self.num_links), end_only=False)
        joint_positions1 = torch.stack([fk1[f'link{i+1}'].get_matrix()[:, :3, 3] for i in range(self.num_links+1)], dim=-2)
        joint_positions2 = torch.stack([fk2[f'link{i+1}'].get_matrix()[:, :3, 3] for i in range(self.num_links+1)], dim=-2)
        
        distances = line_segment_distances(joint_positions1, joint_positions2)
        min_distances = torch.min(distances, dim=-1, keepdim=True).values
        
        boundary_condition = min_distances - self.collision_R
        boundary_condition = boundary_condition.view(original_shape + (-1,))
            
        return boundary_condition
    
    
    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        states = model_input['coords'][...,1:]
        if self.use_symmetry:
            symmetry_mask = states[..., -1] < 0.
            temp = model_input['coords'][symmetry_mask]
            temp[...,1:] *= -1
            model_input['coords'][symmetry_mask] = temp
                
        if self.provide_initial_condition:
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            coords = coords_org
            states = coords[...,1:]
            boundary_condition = self.boundary_condition(states)
            
            output = boundary_condition +  self.net(coords)            
            return {'model_in': coords_org, 'model_out': output}
        else:
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            coords = coords_org
            
            output = self.net(coords)
            return {'model_in': coords_org, 'model_out': output}

        
class Air3D_SymmetryICNet(SingleBVPNet):
    def __init__(self, out_features=1, type='sine', in_features=4, mode='mlp', hidden_features=256, num_hidden_layers=3, 
                 omega_max=3.0, velocity=0.75, angle_alpha=1.2, collision_R=0.25, use_symmetry=False,
                 provide_initial_condition=False,
                 **kwargs):
        super().__init__(out_features, type, in_features, mode, hidden_features, num_hidden_layers, **kwargs)
        self.omega_max = omega_max
        self.velocity = velocity
        self.angle_alpha = angle_alpha
        self.collision_R = collision_R
        
        self.provide_initial_condition = provide_initial_condition
        self.use_symmetry = use_symmetry
    
    def forward(self, model_input, params=None):
        states = model_input['coords'][...,1:]
        if self.use_symmetry:
            symmetry_mask = states[..., -1] < 0.
            temp = model_input['coords'][symmetry_mask]
            temp[...,2:] *= -1
            model_input['coords'][symmetry_mask] = temp
        
        if self.provide_initial_condition:
            if params is None:
                params = OrderedDict(self.named_parameters())

            # Enables us to compute gradients w.r.t. coordinates
            # 
                
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            coords = coords_org
            
            # normalize the value function
            norm_to = 0.02
            mean = 0.25
            var = 0.5
            boundary_condition = torch.linalg.norm(coords[..., 1:3], dim=-1, keepdim=True) - self.collision_R
            boundary_condition = (boundary_condition - mean) * norm_to / var
            output = boundary_condition + self.net(coords)

            return {'model_in': coords_org, 'model_out': output}
        else:
            if params is None:
                params = OrderedDict(self.named_parameters())

            # Enables us to compute gradients w.r.t. coordinates
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            coords = coords_org

            output = self.net(coords)
            return {'model_in': coords_org, 'model_out': output}


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
