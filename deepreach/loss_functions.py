import torch
import torch.nn.functional as F

import diff_operators
import modules, utils

import math
import numpy as np


def initialize_hji_air3D(dataset, minWith, use_symmetry=False):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    alpha_angle = dataset.alpha_angle
    
    def hji_air3D_loss(model_output, gt, validation=False):
        if validation:
            return hji_air3D_validation(model_output, gt)
        else:
            return hji_air3D(model_output, gt)
    
    def hji_air3D_validation(model_output, gt):
        predicted_y = model_output['model_out']
        
        norm_to = 0.02
        mean = 0.25
        var = 0.5
        predicted_y = (predicted_y * var / norm_to) + mean

        loss = torch.sum(torch.abs(gt.view(-1) - predicted_y.view(-1))).item()
        
        return loss

    def hji_air3D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        x_theta = x[..., 3] * 1.0

        # Scale the costate for theta appropriately to align with the range of [-pi, pi]
        dudx[..., 2] = dudx[..., 2] / alpha_angle
        # Scale the coordinates
        x_theta = alpha_angle * x_theta

        # Air3D dynamics
        # \dot x    = -v_a + v_b \cos \psi + a y
        # \dot y    = v_b \sin \psi - a x
        # \dot \psi = b - a

        # Compute the hamiltonian for the ego vehicle
        ham = omega_max * torch.abs(dudx[..., 0] * x[..., 2] - dudx[..., 1] * x[..., 1] - dudx[..., 2])  # Control component
        ham = ham - omega_max * torch.abs(dudx[..., 2])  # Disturbance component
        ham = ham + (velocity * (torch.cos(x_theta) - 1.0) * dudx[..., 0]) + (velocity * torch.sin(x_theta) * dudx[..., 1])  # Constant component

        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet) * 40,
                'diff_constraint_hom': torch.abs(diff_constraint_hom)}

    return hji_air3D_loss


def initialize_hji_pointPlane(dataset, minWith, use_symmetry=False):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    max_velocity_u = dataset.max_velocity_u
    max_velocity_d = dataset.max_velocity_d
    
    def hji_pointPlane_loss(model_output, gt, validation=False):
        if validation:
            return hji_pointPlane_validation(model_output, gt)
        else:
            return hji_pointPlane(model_output, gt)
    
    def hji_pointPlane_validation(model_output, gt):
        predicted_y = model_output['model_out']
        loss = torch.sum(torch.abs(gt.view(-1) - predicted_y.view(-1))).item()
        return loss

    def hji_pointPlane(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            # Compute the hamiltonian
            ham = max_velocity_u * torch.abs(dudx[..., 0] + dudx[..., 1])  # Control component
            ham = ham - max_velocity_d * torch.abs(dudx[..., 2] + dudx[..., 3])  # Disturbance component

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet) * 40,
                'diff_constraint_hom': torch.abs(diff_constraint_hom)}
    

    return hji_pointPlane_loss


def initialize_hji_simpleArm(dataset, minWith, use_symmetry=False):
    # Initialize the loss function for the simpleArm problem
    # The dynamics parameters
    max_joint_velocity = dataset.max_joint_velocity
    num_links = dataset.num_links
    
    def hji_simpleArm_loss(model_output, gt, validation=False):
        if validation:
            return hji_simpleArm_validation(model_output, gt)
        else:
            return hji_simpleArm(model_output, gt)
        
    def hji_simpleArm_validation(model_output, gt):
        predicted_y = model_output['model_out']
        loss = torch.sum(torch.abs(gt.view(-1) - predicted_y.view(-1))).item()
        
        return loss

    def hji_simpleArm(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]
        
        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]
            dudx[torch.isnan(dudx)] = 0.
            dudx1 = dudx[..., :num_links]
            dudx2 = dudx[..., num_links:]

            # Compute the hamiltonian
            ham = max_joint_velocity * torch.abs(torch.sum(dudx1, dim=-1))  # Control component
            ham = ham - max_joint_velocity * torch.abs(torch.sum(dudx2, dim=-1))  # Disturbance component

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)
            
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet) * 40,
                'diff_constraint_hom': torch.abs(diff_constraint_hom)}
    

    return hji_simpleArm_loss


def initialize_hji_NN(dataset, minWith, use_symmetry=False):
    # Initialize the loss function for the NN two realistic arms problem
    # The dynamics parameters
    max_joint_velocity = dataset.max_joint_velocity
    num_links = dataset.num_links

    def hji_nn(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 12)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]
        
        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]
            dudx1 = dudx[..., :num_links // 2]
            dudx2 = dudx[..., num_links // 2:]

            # Compute the hamiltonian
            ham = max_joint_velocity * torch.abs(torch.sum(dudx1, dim=-1))  # Control component
            ham = ham - max_joint_velocity * torch.abs(torch.sum(dudx2, dim=-1))  # Disturbance component

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)
            
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet) * 40,
                'diff_constraint_hom': torch.abs(diff_constraint_hom)}
    

    return hji_nn