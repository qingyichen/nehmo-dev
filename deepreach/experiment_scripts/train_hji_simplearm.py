# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./hji_logs/logs_simplearm', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--num_links', type=int, default=2)
p.add_argument('--robot1_urdf_path', type=str, default='')
p.add_argument('--robot2_urdf_path', type=str, default='')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=400,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=0.6, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=2000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=-1, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples at each time step')

p.add_argument('--max_joint_velocity', type=float, default=0.5, required=False, help='Max joint velocity of the arm')
p.add_argument('--collisionR', type=float, default=0.1, required=False, help='Collision radius between arms')
p.add_argument('--minWith', type=str, default='zero', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')
p.add_argument('--use_symmetry', action='store_true', default=False, required=False, help='Whether to compute symmetric loss during training')
p.add_argument('--provide_initial_condition', action='store_true', default=False, required=False, help='Whether to provide the initial condition to the model')

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# logging dir adjustments
opt.logging_root = opt.logging_root + f"_{opt.num_links}links"
opt.experiment_name = opt.experiment_name + f"_seed{opt.seed}"
if opt.robot1_urdf_path == '' or opt.robot2_urdf_path == '':
  opt.robot1_urdf_path = f'envs/arm_urdfs/simple_{opt.num_links}link_arm1.urdf'
  opt.robot2_urdf_path = f'envs/arm_urdfs/simple_{opt.num_links}link_arm2.urdf'


# Set the source coordinates for the target set and the obstacle sets
source_coords = [0., 0., 0.]
if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs


torch.manual_seed(opt.seed)
dataset = dataio.ReachabilitySimpleArmSource(numpoints=65000, collisionR=opt.collisionR, max_joint_velocity=opt.max_joint_velocity,
                                          pretrain=opt.pretrain, tMin=opt.tMin, num_links=opt.num_links,
                                          tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                          pretrain_iters=opt.pretrain_iters, seed=opt.seed,
                                          num_src_samples=opt.num_src_samples,
                                          robot1_urdf_path=opt.robot1_urdf_path, robot2_urdf_path=opt.robot2_urdf_path,
                                          use_symmetry=opt.use_symmetry)
val_dataset = dataio.ReachabilitySimpleArmValidationSet(validation_data_path='./deepreach/validation_sets/simpleArm.pkl')
dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=2048, pin_memory=True, num_workers=0)

model = modules.SimpleArm_SymmetryICNet(in_features=opt.num_links*2+1, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl,
                             robot1_urdf_path=opt.robot1_urdf_path, robot2_urdf_path=opt.robot2_urdf_path,
                             max_joint_velocity=opt.max_joint_velocity, collisionR=opt.collisionR, num_links=opt.num_links,
                             use_symmetry=opt.use_symmetry, provide_initial_condition=opt.provide_initial_condition)
model.cuda()

# Define the loss
loss_fn = loss_functions.initialize_hji_simpleArm(dataset, opt.minWith, opt.use_symmetry)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, val_dataloader=val_dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
               use_lbfgs=opt.use_lbfgs, validation_fn=None, start_epoch=opt.checkpoint_toload,
               training_setting=opt) # TODO: may add validation fn and validation dataloader
