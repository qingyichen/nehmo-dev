import os
import torch
import pickle
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, n_links=12, num_hidden_layers=8, hidden_size=1024, softplus_beta=1.):
        super().__init__()
        self.input_size = n_links
        self.num_layers = num_hidden_layers + 2
        if softplus_beta > 0:
            self.activation_layer = nn.Softplus(beta=softplus_beta)
        else:
            self.activation_layer = nn.ReLU()

        self.linear0 = nn.Linear(self.input_size, hidden_size)
        for layer_id in range(1, num_hidden_layers + 1):
            if layer_id == num_hidden_layers // 2:
                output_dim = hidden_size - self.input_size
            else:
                output_dim = hidden_size
            setattr(self, f'linear{layer_id}', nn.Linear(hidden_size, output_dim))
        setattr(self, f'linear{num_hidden_layers+1}', nn.Linear(hidden_size, 1))
    
    def forward(self, qpos):
        outputs = qpos
        for i in range(self.num_layers):
            layer = getattr(self, f"linear{i}")
            if i == self.num_layers // 2:
                outputs = torch.cat([outputs, qpos], dim=-1)
            outputs = layer(outputs)
            if i != self.num_layers - 1:
                outputs = self.activation_layer(outputs)
        return outputs


class DualArmDistanceDataset(Dataset):
    def __init__(self, dataset_file, dataset_dir) -> Dataset:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.data_file = os.path.join(self.dataset_dir, dataset_file)
        with open(self.data_file, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return self.data['qpos'].shape[0]
    
    def __getitem__(self, index):
        qpos = self.data['qpos'][index]
        s = self.data['distances'][index]
        
        return qpos, s
    

def make_model(params):
    model = MLP(
            n_links=params.n_links, 
            num_hidden_layers=params.num_hidden_layers, 
            hidden_size=params.hidden_size,
            softplus_beta=params.softplus_beta,
        )
    model_name = f"armMlp{params.num_hidden_layers}x{params.hidden_size}"    

    return model, model_name


def make_datasets(params):
    dataset = DualArmDistanceDataset(
        dataset_dir=params.dataset_dir, 
        dataset_file=params.data,
    )
    return dataset


def distance_and_collision_loss(pred_distances, gt_distances):
    collision_mask = gt_distances <= 0.
    non_collision_mask = torch.logical_not(collision_mask)
    distance_loss = torch.square(pred_distances[non_collision_mask] - gt_distances[non_collision_mask]).mean()
    collision_loss = torch.clamp(pred_distances[collision_mask], min=0.).square().mean()
    
    loss_dict = {
        'distance_loss': distance_loss, 
        'collision_loss': collision_loss
    }
    
    return loss_dict


def make_dataloaders(params, training_proportion=0.8):
    dataset = make_datasets(params)
        
    training_size = int(len(dataset) * training_proportion)
    training_dataset, val_dataset = torch.utils.data.random_split(dataset, [training_size, len(dataset)-training_size])
    training_dataloader = DataLoader(training_dataset, batch_size=params.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True)
    dataloaders_dict = {'train':training_dataloader, 'val':val_dataloader}
    print(f"Data size: training {len(training_dataset)}, validation {len(val_dataset)}")

    return dataloaders_dict