import os
import torch
import pickle
import matplotlib.pyplot as plt

def combine_datasets(num_sub_datasets = 80, num_data = 32000, dataset_dir = 'UR5_datasets_and_training/UR5_d0.6_mesh_distances_dataset'):
    alldataset_qpos = torch.zeros((num_sub_datasets * num_data, 6))
    alldataset_distances = torch.zeros(num_sub_datasets * num_data)

    for i_dataset in range(num_sub_datasets):
        filename = f'seed{i_dataset}num_data{num_data}.pkl'
        with open(os.path.join(dataset_dir, filename), 'rb') as f:
            data = pickle.load(f)
            print(data['qpos'][0])
            alldataset_qpos[i_dataset * num_data: (i_dataset+1) * num_data] = data['qpos']
            alldataset_distances[i_dataset * num_data: (i_dataset+1) * num_data] = data['distances']
    with open(os.path.join(dataset_dir, f'seed{0}-{num_sub_datasets-1}num_data{num_sub_datasets * num_data}.pkl'), 'wb') as f:
        data = {
            'qpos': alldataset_qpos,
            'distances': alldataset_distances,
        }
        pickle.dump(data, f)
        
    print("Finished combining sub-datasets.")
    
def distances_histograms(dataset_filename='UR5_datasets_and_training/UR5_d0.6_mesh_distances_dataset/seed0-79num_data2560000.pkl'):
    with open(dataset_filename, 'rb') as f:
        data = pickle.load(f)
        distances = data['distances']
    plt.hist(distances)
    plt.xlabel('Distances [m]')
    # plt.ylabel('Number of Occurrences')
    plt.savefig('distance_hist.png', dpi=600)
        

if __name__ == '__main__':
    combine_datasets()
    # distances_histograms()
    
    
    
    