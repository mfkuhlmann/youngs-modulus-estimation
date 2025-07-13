import numpy as np
import os

def conv2D_output_size(img_size, padding, kernel_size, stride):
	output_shape=(np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
				  np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
	return output_shape

def conv3D_output_size(img_size, padding, kernel_size, stride):
	output_shape=(np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
				  np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
				  np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
	return output_shape 

def list_files(folder_path, file_paths, config):
    # Iterate through the list
    for item in os.listdir(folder_path):
        item_path = f'{folder_path}/{item}'
        marker_signal = '_markers' if config['use_markers'] else '.pkl'
        if os.path.isfile(item_path) and item.count(config['img_style']) > 0 and item.count(marker_signal) > 0:
            file_paths.append(item_path)
        elif os.path.isdir(item_path):
            list_files(item_path, file_paths, config)
