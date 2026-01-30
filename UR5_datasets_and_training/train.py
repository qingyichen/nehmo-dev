import os
import copy
import torch
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from training_utils import make_model, make_dataloaders, distance_and_collision_loss

import json
from datetime import datetime

def read_params():
    parser = argparse.ArgumentParser(description="SDF RTD Training")
    # general env setting
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot", type=str, default='UR5')
    parser.add_argument('--n_links', type=int, default=12)

    # dataset setting
    parser.add_argument('--dataset_dir', type=str, default='UR5_datasets_and_training')
    parser.add_argument('--data', type=str, default='UR5_d0.6_mesh_distances_dataset/seed0-79num_data2560000.pkl')

    # model setting
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--softplus_beta', type=float, default=1)

    # learning setting
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument("--device", type=str, default='cuda')

    return parser.parse_args()

def train_model(model, dataloaders, optimizer, device, summaries_dir, num_epochs=80):
    model = model.to(device)
    best_loss = torch.inf
    best_model = None
    if not os.path.exists(os.path.join(summaries_dir,'saved_models')):
        os.mkdir(os.path.join(summaries_dir,'saved_models'))
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            loss_dict = {}
            running_loss_dict = {}
            max_difference = torch.tensor(0.0).to(device)
            max_collision_prediction = torch.tensor(0.0).to(device)
            mean_collision_prediction = torch.tensor(0.0).to(device)
            mean_difference = torch.tensor(0.0).to(device)
            num_non_collision_data = 0
            num_collision_data = 0

            # Iterate over data.
            for (qpos, labels) in tqdm(dataloaders[phase]):
                qpos, labels = qpos.to(device), labels.to(device)
                labels = labels.view(-1,1)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(qpos=qpos)
                    
                    loss_dict = distance_and_collision_loss(outputs, labels)
                    loss = 0.0
                    for k, v in loss_dict.items():
                        loss += v

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                for k, v in loss_dict.items():  
                    curr_loss = v.item()
                    if k in running_loss_dict:
                        running_loss_dict[k] += curr_loss
                    else:
                        running_loss_dict[k] = curr_loss
                outputs_data = outputs.detach()
                collision_mask = (labels <= 0.0)
                num_collision_data += collision_mask.sum()
                non_collision_mask = torch.logical_not(collision_mask)
                num_non_collision_data += non_collision_mask.sum()              
                if torch.any(collision_mask):
                    max_collision_prediction = torch.max(torch.max(outputs_data[collision_mask]), max_collision_prediction)
                mean_collision_prediction += outputs_data[collision_mask].sum()
                    
                outputs_data = torch.clamp(outputs_data, min=0.)
                curr_difference = torch.abs(outputs_data[non_collision_mask] - labels[non_collision_mask])
                mean_difference += curr_difference.sum()
                max_difference = torch.max(torch.max(curr_difference), max_difference)

            log = {
                f'{phase}/max pred vs. truth difference': max_difference,
                f'{phase}/max pred for collision states': max_collision_prediction,
                f'{phase}/mean pred vs. truth difference': mean_difference / num_non_collision_data,
                f'{phase}/mean pred for collision states': mean_collision_prediction / num_collision_data,
            }
            
            epoch_loss = 0.0
            for k, v in running_loss_dict.items():
                curr_loss = v / len(dataloaders[phase])
                log[f'{phase}/{k}'] = curr_loss
                epoch_loss += curr_loss
            for k, v in log.items(): 
                writer.add_scalar(k + " vs. epochs", v, epoch)

            print(f'{phase} Loss: {epoch_loss}')
            if phase == 'val' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and (epoch - 1) % (num_epochs // 4) == 0:
                torch.save(model.state_dict(), os.path.join(summaries_dir, 'saved_models', f'model_epoch{epoch}_loss{epoch_loss:.7f}.pth'))

    print(f'Best val Loss: {best_loss}')
    torch.save(best_model, os.path.join(summaries_dir, 'saved_models', f'best_model_loss{best_loss:.9f}.pth'))

    return best_model, best_loss


if __name__ == '__main__':
    params = read_params()
    torch.manual_seed(params.seed)

    # Data
    dataloaders_dict = make_dataloaders(params, training_proportion=0.8)
    
    # Model
    model, model_name = make_model(params)

    # Logistics
    now = datetime.now()
    since = now.strftime("%m-%d-%H-%M")
    experiment_name = f"{params.robot}_{model_name}_lr{params.lr}_{since}"
    summaries_dir = os.path.join(params.dataset_dir, params.robot, experiment_name)
    writer = SummaryWriter(summaries_dir)
        
    with open(os.path.join(summaries_dir, "training_setting.json"), 'w') as f:
        json.dump(params.__dict__, f, indent=4)
    print(f"Launching experiment with config {params.__dict__}")
    device = torch.device(f"{params.device}")
    print(f"Starting experiment {experiment_name} using device {device}")

    # Learning
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, betas=(params.beta1, params.beta2), weight_decay=params.weight_decay)
    best_model, best_loss = train_model(model, dataloaders_dict, optimizer, num_epochs=params.num_epochs, device=device, summaries_dir=summaries_dir)

    print(f"Training ends with best loss = {best_loss}")
