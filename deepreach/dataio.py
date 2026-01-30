import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.io as spio
import torch
from torch.utils.data import Dataset

import utils
import pickle
import pytorch_kinematics as pk
from utils import line_segment_distances
from torch.autograd import grad


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


class ReachabilityNNSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.0, max_joint_velocity=0.5, num_links=12,
        bc_model=None, device='cpu', pretrain=False, tMin=0.0, tMax=0.6, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, num_src_samples=1000, seed=0, use_symmetry=False):
        super().__init__()
        torch.manual_seed(0)
        
        self.bc_model = bc_model.to(device)
        self.device = device

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.max_joint_velocity = max_joint_velocity
        self.collisionR = collisionR

        self.num_states = num_links
        self.num_links = num_links

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.use_symmetry = use_symmetry
        
        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states, device=self.device).uniform_(-1, 1) * torch.pi
        
        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1, device=self.device) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1, device=self.device).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time
        
        if self.use_symmetry:
            coords = coords[torch.logical_and(coords[..., 7] >= -torch.pi / 2, coords[..., 7] <= torch.pi / 2)]

        with torch.no_grad():
            states = coords[..., 1:]
            boundary_values = self.bc_model(states)

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False
        model_input = {'coords': coords}

        return model_input, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
    

class ReachabilitySimpleArmSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.1, max_joint_velocity=0.5, num_links=2,
        robot1_urdf_path='', robot2_urdf_path='',
        pretrain=False, tMin=0.0, tMax=0.6, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, num_src_samples=1000, seed=0, use_symmetry=False):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.max_joint_velocity = max_joint_velocity
        self.collisionR = collisionR

        self.num_states = num_links * 2
        self.num_links = num_links

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.use_symmetry = use_symmetry
        
        self.chain1 = pk.build_serial_chain_from_urdf(open(robot1_urdf_path).read(), f"link{self.num_links+1}")
        self.chain2 = pk.build_serial_chain_from_urdf(open(robot2_urdf_path).read(), f"link{self.num_links+1}")

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1) * torch.pi
        
        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time
        
        if self.use_symmetry:
            coords = coords[coords[..., -1] >= 0.]
            
        with torch.no_grad():
            states = coords[..., 1:]
            # set up the initial value function
            q1 = states[..., :self.num_links]
            q2 = states[..., self.num_links:]
            
            fk1 = self.chain1.forward_kinematics(q1, end_only=False)
            fk2 = self.chain2.forward_kinematics(q2, end_only=False)
            joint_positions1 = torch.stack([fk1[f'link{i+1}'].get_matrix()[:, :3, 3] for i in range(self.num_links+1)], dim=-2)
            joint_positions2 = torch.stack([fk2[f'link{i+1}'].get_matrix()[:, :3, 3] for i in range(self.num_links+1)], dim=-2)

            distances = line_segment_distances(joint_positions1, joint_positions2)
            boundary_values = torch.min(distances, dim=-1, keepdim=True).values - self.collisionR

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False
        model_input = {'coords': coords}

        return model_input, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


class ReachabilitySimpleArmValidationSet(Dataset):
    def __init__(self, validation_data_path='./deepreach/validation_sets/simpleArm.pkl'):
        super().__init__()
        with open(validation_data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.data['simpleArm_in'] = self.data['simpleArm_in'].float().view(-1,4)
        self.data['simpleArm_in'] = torch.cat([torch.ones_like(self.data['simpleArm_in'][:,0:1]) * 0.5, self.data['simpleArm_in']], dim=1)
        self.data['simpleArm_out'] = self.data['simpleArm_out'].float().view(-1)

    def __len__(self):
        return self.data['simpleArm_in'].shape[0]

    def __getitem__(self, idx):
        gt = self.data['simpleArm_out'][idx]
        
        return {'coords': self.data['simpleArm_in'][idx]}, gt


class ReachabilityAir3DSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.25, velocity=0.6, omega_max=1.1, threshold=0.0,
        pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, seed=0, use_symmetry=False):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.omega_max = omega_max
        self.collisionR = collisionR

        self.alpha_angle = angle_alpha * math.pi

        self.num_states = 3

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.use_symmetry = use_symmetry

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        
        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time
        
        if self.use_symmetry:
            coords = coords[coords[..., -1] >= 0.]
            
        # set up the initial value function
        boundary_values = torch.norm(coords[:, 1:3], dim=1, keepdim=True) - self.collisionR

        # normalize the value function
        norm_to = 0.02
        mean = 0.25
        var = 0.5

        boundary_values = (boundary_values - mean)*norm_to/var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False
        model_input = {'coords': coords}

        return model_input, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
    
    
class ReachabilityAir3DValidationSet(Dataset):
    def __init__(self, validation_data_path='./deepreach/validation_sets/air3D_validation_set.pkl'):
        super().__init__()
        with open(validation_data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return self.data['air3D_in'].shape[0]

    def __getitem__(self, idx):
        gt = self.data['air3D_out'][idx]
        
        return {'coords': self.data['air3D_in'][idx]}, gt


class ReachabilityPointPlaneSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.2, velocity=0.5, pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, num_src_samples=1000, seed=0, bound=2.0, validation=False):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.max_velocity_u = velocity
        self.max_velocity_d = velocity
        self.collisionR = collisionR
        self.num_states = 4
        self.bound = bound

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.validation = validation

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-self.bound, self.bound)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # set up the initial value function
        boundary_values = torch.norm(coords[..., 1:3] - coords[..., 3:5], dim=-1, keepdim=True) - self.collisionR
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False
        model_input = {'coords': coords}

        if not self.validation:
            return model_input, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
        else:
            return model_input, boundary_values
    
    