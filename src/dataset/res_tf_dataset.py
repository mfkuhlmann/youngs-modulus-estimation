from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pickle
import os
import random

N_FRAMES = 3
WARPED_CROPPED_IMG_SIZE = (250, 350)

class CustomDataset(Dataset):
    def __init__(self, config, paths_to_files, labels, normalization_values, \
                 validation_dataset=False,
                 frame_tensor=torch.zeros((N_FRAMES, 3, WARPED_CROPPED_IMG_SIZE[0], WARPED_CROPPED_IMG_SIZE[1])),
                 force_tensor=torch.zeros((N_FRAMES, 1)),
                 width_tensor=torch.zeros((N_FRAMES, 1)),
                 estimation_tensor=torch.zeros((2, 1)),
                 label_tensor=torch.zeros((1, 1))
        ):
        # Data parameters 
        self.data_dir               = config['data_dir']
        self.training_data_folder   = config['training_data_folder']
        self.n_frames               = config['n_frames']
        self.img_size               = config['img_size']
        self.img_style              = config['img_style']
        self.n_channels             = config['n_channels']
        self.rubber_only            = config['rubber_only']
        self.val_on_seen_objects    = config['val_on_seen_objects']
        self.use_markers            = config['use_markers']
        self.use_force              = config['use_force']
        self.use_width              = config['use_width']
        self.use_estimations         = config['use_estimations']
        self.use_transformations    = config['use_transformations']
        self.use_width_transforms   = config['use_width_transforms']
        self.exclude                = config['exclude']

        # Define training parameters
        self.epochs             = config['epochs']
        self.batch_size         = config['batch_size']
        self.img_feature_size   = config['img_feature_size']
        self.fwe_feature_size   = config['fwe_feature_size']
        self.val_pct            = config['val_pct']
        self.learning_rate      = config['learning_rate']
        self.gamma              = config['gamma']
        self.random_state       = config['random_state']
        
        
        self.embedding_dim = config['embedding_dim']
        self.layer = config['layer']
        self.head = config['head']
        self.output_dim = config['output_dim']
        self.decoder_layer_nodes = config['decoder_layer_nodes']
        self.estimation_decoder_layers = config['estimation_decoder_layers']
    
    
        self.validation_dataset = validation_dataset
        self.normalization_values = normalization_values
        self.input_paths = paths_to_files
        self.normalized_modulus_labels = labels

        if self.use_width_transforms:
            self.input_paths = 2*self.input_paths
            self.normalized_modulus_labels = 2*self.normalized_modulus_labels
            self.noise_force = [ i > len(self.input_paths)/2 and i % 2 == 1 for i in range(len(self.input_paths)) ]
            self.noise_width = [ i > len(self.input_paths)/2 for i in range(len(self.input_paths)) ]

        # Define attributes to use to conserve memory
        self.base_name      = ''
        self.x_frames       = frame_tensor
        self.x_forces       = force_tensor
        self.x_widths       = width_tensor
        self.x_estimations  = estimation_tensor
        self.y_label        = label_tensor
    
    def __len__(self):
        return len(self.normalized_modulus_labels)
    
    def __getitem__(self, idx):
        self.x_frames       = self.x_frames.zero_()
        self.x_forces       = self.x_forces.zero_()
        self.x_widths       = self.x_widths.zero_()
        self.x_estimations  = self.x_estimations.zero_()
        self.y_label        = self.y_label.zero_()

        folders = self.input_paths[idx].split('/')
        object_name = folders[folders.index(self.training_data_folder) + 1]

        # Read and store frames in the tensor
        with open(self.input_paths[idx], 'rb') as file:
            if self.img_style == 'depth':
                self.x_frames[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(3).permute(0, 3, 1, 2)
                self.x_frames /= self.normalization_values['max_depth']
            else:
                self.x_frames[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).permute(0, 3, 1, 2)

        # Unpack force measurements
        self.base_name = os.path.dirname(self.input_paths[idx])
        if self.use_force:
            with open(self.base_name + '/forces.pkl', 'rb') as file:
                self.x_forces[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
                self.x_forces /= self.normalization_values['max_force']

        # Unpack gripper width measurements
        if self.use_width:
            with open(self.base_name + '/widths.pkl', 'rb') as file:
                self.x_widths[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
                self.x_widths[:] /= self.normalization_values['max_width']

            if self.use_width_transforms:
                if self.noise_width[idx]:
                    noise_amplitude = min(
                        1 - self.x_widths.max(),
                        self.x_widths.min()
                    )
                    if random.random() > 0.5:
                        self.x_widths += noise_amplitude * random.random()
                    else:
                        self.x_widths -= noise_amplitude * random.random()
        
        # Unpack modulus estimations
        if self.use_estimations:
            with open(self.base_name + '/elastic_estimate.pkl', 'rb') as file:
                self.x_estimations[0] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
            with open(self.base_name + '/hertz_estimate.pkl', 'rb') as file:
                self.x_estimations[1] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
        
        # Unpack label
        self.y_label[0] = self.normalized_modulus_labels[idx]

        return self.x_frames.clone(), self.x_forces.clone(), self.x_widths.clone(), self.x_estimations.clone(), self.y_label.clone(), object_name