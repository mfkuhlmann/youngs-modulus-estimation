
import numpy as np
import torch

import torchvision
import torch.nn as nn
import lightning

from model.VGG_LSTM.triple_layer import TripleLSTMModel

from torch.optim import lr_scheduler

N_FRAMES = 3

class ModulusModel(lightning.LightningModule):
    def __init__(self, config):
        super(ModulusModel, self).__init__()
        self.config = config
        self.save_hyperparameters(config)


        # Create max values for scaling
        self.normalization_values = { # Based on acquired data maximums
            'min_modulus': 1e3,
            'max_modulus': 1e12,
            'min_estimate': 1e2,
            'max_estimate': 1e14,
            'max_depth': 7.0,
            'max_width': 0.08, # Measured from Franka Panda gripper
            'max_force': 60.0,
        }

        # Reduce prediction range if objects are all rubber
        if self.hparams.rubber_only:
            self.normalization_values['min_modulus'] = 1e5
            self.normalization_values['max_modulus'] = 1e8

        if "shores" in self.hparams.dataset_name:
            print("Loaded Shores normalization values")
            self.normalization_values['min_modulus'] = 1e3
            self.normalization_values['max_modulus'] = 1e7
        
        print(self.normalization_values)

        # Normalize based on mean and std computed over the dataset
        if self.hparams.n_channels == 3:
            # Use the diff mean and std computed for our dataset
            self.image_normalization = torchvision.transforms.Normalize( \
                                            [0.49638007, 0.49770336, 0.49385751], \
                                            [0.04634926, 0.06181679, 0.07152624] \
                                        )
            
                # Apply random flipping transformations
        if self.hparams.use_transformations:
            self.random_transformer = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(0.5),
                    torchvision.transforms.RandomVerticalFlip(0.5),
                ])

        input_dim = self.hparams.input_dim  
        self.video_encoder = TripleLSTMModel(input_dim, self.hparams.scalar1_dim, self.hparams.scalar2_dim, self.hparams.scalar3_dim, self.hparams.scalar4_dim, self.hparams.hidden_dim, self.hparams.lstm_dropout)

        # self.criterion = nn.MSELoss()
        self.params = list(self.video_encoder.parameters())

        if self.hparams.use_log_normalization:
            self.unnormalization_function = self.log_unnormalize
        else:
            self.unnormalization_function = self.unnormalize

    def forward(self, x_frames, x_forces, x_widths, x_estimations, batch_size):
        x_frames = x_frames.view(-1, self.hparams.n_channels, self.hparams.img_size[0], self.hparams.img_size[1])

        # Normalize images
        if self.hparams.n_channels == 3:
            x_frames = self.image_normalization(x_frames)

        # Apply random transformations for training
        if self.hparams.use_transformations:
            x_frames = self.random_transformer(x_frames) # Apply V/H flips
            
        x_frames = x_frames.view(batch_size, self.hparams.n_frames, self.hparams.n_channels, self.hparams.img_size[0], self.hparams.img_size[1])
        
        x_estimations = torch.clamp(x_estimations, min=self.normalization_values['min_estimate'], max=self.normalization_values['max_estimate'])
        x_estimations = self.log_normalize(x_estimations, x_max=self.normalization_values['max_estimate'], x_min=self.normalization_values['min_estimate'], use_torch=True)
  
        
        features = []
        x_forces_list = []
        x_widths_list = []
        for i in range(self.hparams.n_frames):            

            features.append(x_frames[:, i, :, :, :])
        # shape = self.get_list_shape(features)

        
        ###################  change  ########################
        # print(x_forces)
        ###################  FW  ########################
        x_forcess = torch.unbind(x_forces[:, :, :].squeeze(-1), dim=1)
        x_forces_list = [t.view(-1, 1) for t in x_forcess]
        
        x_widthss = torch.unbind(x_widths[:, :, :].squeeze(-1), dim=1)
        # print("x_widthss: ", x_widthss)
        x_widths_list = [t.view(-1, 1) for t in x_widthss]
        ###################  FW  ########################
        
        ###################  E  ########################
        tensor_1_list = []
        tensor_2_list = []
        expanded_tensor = x_estimations.repeat(1, 1, 3)
        tensor_1 = expanded_tensor[:, 0, :] 
        tensor_2 = expanded_tensor[:, 1, :] 
        #print(tensor_1)
        tensor_1s = torch.unbind(tensor_1, dim=1)
        tensor_1_list = [t.view(-1, 1) for t in tensor_1s]
        tensor_2s = torch.unbind(tensor_2, dim=1)
        tensor_2_list = [t.view(-1, 1) for t in tensor_2s]
        # #exit()
        ###################  E  ########################
        
        ###################  change  ########################

        outputs = self.video_encoder(features, x_forces_list, x_widths_list, tensor_1_list, tensor_2_list)
        #outputs = self.video_encoder(features, x_forces_list, x_widths_list, None, None)
        #outputs = self.video_encoder(features, x_forces_list, None, None, None)
        #outputs = self.video_encoder(features, None, None, None, None)

        return outputs
    
    def log_normalize(self, x, x_max=None, x_min=None, use_torch=False):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        
        # print(x)
        # print(self.x_min_cuda)
        # print(self.x_max_cuda)
        
        if use_torch:
            x_max = torch.tensor(x_max, device=x.device)
            x_min = torch.tensor(x_min, device=x.device)

            return (torch.log10(x) - torch.log10(x_min)) / (torch.log10(x_max) - torch.log10(x_min))
        return (np.log10(x) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
    
    def normalize(self, x, x_max=None, x_min=None):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        return (x - x_min) / (x_max - x_min)
    
    def unnormalize(self, x_normal, x_max=None, x_min=None):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        return x_min + x_normal * (x_max - x_min)

    def log_unnormalize(self, x_normal, x_max=None, x_min=None):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        return x_min * (x_max/x_min)**(x_normal)
    
    def loss(self, outputs, y):

        loss = nn.functional.mse_loss(outputs.squeeze(), y.squeeze())

        # Add regularization to loss
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.params:
            l2_reg += torch.norm(param)
        loss += 0.00005 * l2_reg
        return loss
    
    def training_step(self, batch, batch_idx):

        x_frames, x_forces, x_widths, x_estimations, y, _ = batch
        batch_size = x_frames.shape[0]
        y = y.squeeze()
        outputs = self.forward(x_frames, x_forces, x_widths, x_estimations, x_frames.shape[0])
        outputs = outputs.squeeze()
        loss = self.loss(outputs, y)

        abs_log_diff = torch.abs(torch.log10(self.unnormalization_function(outputs.cpu())) - torch.log10(self.unnormalization_function(y.cpu()))).detach()
        abs_diff = torch.abs(self.unnormalization_function(outputs.cpu()) - self.unnormalization_function(y.cpu())).detach()
        abs_acc = torch.clip((self.unnormalization_function(y.cpu()).detach() - abs_diff), min=0.0) / self.unnormalization_function(y.cpu()).detach()
        stats = {
            'train_loss': loss,
            'train_absolute_diff': abs_diff.mean(),
            'train_absolute_diff_min': abs_diff.min(), 
            'train_absolute_diff_max': abs_diff.max(),
            'train_absolute_acc': abs_acc.mean(),
            'train_log_acc': (abs_log_diff <= 1).sum()/batch_size, # How often the prediction is within 10x of the true value
            'train_avg_log_diff': abs_log_diff.mean(),
            'train_n_mse': nn.functional.mse_loss(outputs.squeeze(), y.squeeze())
            # 'lr': self.lr_schedulers().get_last_lr()[0]
        }
        self.log_dict(stats, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x_frames, x_forces, x_widths, x_estimations, y, _ = batch
        batch_size = x_frames.shape[0]
        y = y.squeeze()
        outputs = self.forward(x_frames, x_forces, x_widths, x_estimations, x_frames.shape[0])
        outputs = outputs.squeeze()
        loss = self.loss(outputs, y)

        abs_log_diff = torch.abs(torch.log10(self.unnormalization_function(outputs.cpu())) - torch.log10(self.unnormalization_function(y.cpu()))).detach()
        abs_diff = torch.abs(self.unnormalization_function(outputs.cpu()) - self.unnormalization_function(y.cpu())).detach()
        abs_acc = torch.clip((self.unnormalization_function(y.cpu()).detach() - abs_diff), min=0.0) / self.unnormalization_function(y.cpu()).detach()
        stats = {
            'val_loss': loss,
            'val_absolute_diff': abs_diff.mean(),
            'val_absolute_diff_min': abs_diff.min(), 
            'val_absolute_diff_max': abs_diff.max(),
            'val_absolute_acc': abs_acc.mean(),
            'val_log_acc': (abs_log_diff <= 1).sum()/batch_size, # How often the prediction is within 10x of the true value
            'val_avg_log_diff': abs_log_diff.mean(),
            'val_n_mse': nn.functional.mse_loss(outputs.squeeze(), y.squeeze())
        }
        self.log_dict(stats, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):

        x_frames, x_forces, x_widths, x_estimations, y, _ = batch
        batch_size = x_frames.shape[0]
        y = y.squeeze()
        outputs = self.forward(x_frames, x_forces, x_widths, x_estimations, x_frames.shape[0])
        outputs = outputs.squeeze()
        loss = self.loss(outputs, y)

        abs_log_diff = torch.abs(torch.log10(self.unnormalization_function(outputs.cpu())) - torch.log10(self.unnormalization_function(y.cpu()))).detach()
        abs_diff = torch.abs(self.unnormalization_function(outputs.cpu()) - self.unnormalization_function(y.cpu())).detach()
        abs_acc = torch.clip((self.unnormalization_function(y.cpu()).detach() - abs_diff), min=0.0) / self.unnormalization_function(y.cpu()).detach()

        stats = {
            'test_loss': loss,
            'test_absolute_diff': abs_diff.mean(),
            'test_absolute_diff_min': abs_diff.min(), 
            'test_absolute_diff_max': abs_diff.max(),
            'test_absolute_acc': abs_acc.mean(),
            'test_log_acc': (abs_log_diff <= 1).sum()/batch_size, # How often the prediction is within 10x of the true value
            'test_avg_log_diff': abs_log_diff.mean(),
            'test_n_mse': nn.functional.mse_loss(outputs.squeeze(), y.squeeze())
        }

        self.log_dict(stats, batch_size=batch_size)
        return loss

    def configure_optimizers(self,):
        optimizer = torch.optim.Adam(self.params, lr=self.hparams.learning_rate)
        if self.hparams.gamma is not None:
            scheduler  = lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.gamma)
            return [optimizer], [scheduler]
        else:
            return optimizer
    