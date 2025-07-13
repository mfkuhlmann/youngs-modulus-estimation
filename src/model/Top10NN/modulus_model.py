import lightning
from torch.optim import lr_scheduler


import numpy as np

import torch
import torch.nn as nn
from model.Top10NN.nn_modules import EncoderCNN, ForceFC, WidthFC, EstimationDecoderFC, DecoderFC

import torchvision



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
            self.normalization_values['min_modulus'] = 1e3
            self.normalization_values['max_modulus'] = 1e9
        
        if "shores" in self.hparams.dataset_name:
            # self.normalization_values['min_modulus'] = 1e5
            self.normalization_values['max_modulus'] = 1e7


        # Sub-Models for input
        self.video_encoder = EncoderCNN(
            img_x=self.hparams.img_size[0],
            img_y=self.hparams.img_size[1],
            input_channels=self.hparams.n_channels,
            CNN_embed_dim=self.hparams.img_feature_size,
            dropout_pct=self.hparams.video_dropout
        )

        self.force_encoder = ForceFC(
                                    input_dim=self.hparams.n_frames,
                                    hidden_size=self.hparams.fwe_feature_size, 
                                    output_dim=self.hparams.fwe_feature_size, 
                                    dropout_pct=self.hparams.force_dropout
                                ) if self.hparams.use_force else None
        
        self.width_encoder = WidthFC(
                                    input_dim=self.hparams.n_frames,
                                    hidden_size=self.hparams.fwe_feature_size,
                                    output_dim=self.hparams.fwe_feature_size,
                                    dropout_pct=self.hparams.width_dropout
                                ) if self.hparams.use_width else None
        
        self.estimation_decoder = EstimationDecoderFC(
                                    input_dim=2 + self.hparams.decoder_output_size,
                                    FC_layer_nodes=self.hparams.est_decoder_size,
                                    output_dim=1,
                                    dropout_pct=self.hparams.est_dropout
                                ) if self.hparams.use_estimations else None

        decoder_input_size = self.hparams.n_frames * self.hparams.img_feature_size
        if self.hparams.use_force:
            decoder_input_size += self.hparams.fwe_feature_size
        if self.hparams.use_width:
            decoder_input_size += self.hparams.fwe_feature_size
        decoder_output_size = self.hparams.decoder_output_size if self.hparams.use_estimations else 1
        self.decoder = DecoderFC(input_dim=decoder_input_size, FC_layer_nodes=self.hparams.decoder_size, output_dim=decoder_output_size, dropout_pct=self.hparams.decoder_dropout)

        self.params = list(self.video_encoder.parameters())
        if self.hparams.use_force: 
            self.params += list(self.force_encoder.parameters())
        if self.hparams.use_width: 
            self.params += list(self.width_encoder.parameters())
        if self.hparams.use_estimations: 
            self.params += list(self.estimation_decoder.parameters())
        self.params += list(self.decoder.parameters())


        # Normalize based on mean and std computed over the dataset
        if self.hparams.n_channels == 3:
            # Use the diff mean and std computed for our dataset
            # TODO: check this!!!!!!!!!!!!!
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
            
        if self.hparams.use_log_normalization:
            self.unnormalization_function = self.log_unnormalize
        else:
            self.unnormalization_function = self.unnormalize

        return
    
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
    
    
    def forward(self, x_frames, x_widths, x_forces, x_estimations, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        x_frames = x_frames.view(-1, self.hparams.n_channels, self.hparams.img_size[0], self.hparams.img_size[1])

        # Normalize images
        if self.hparams.n_channels == 3:
            x_frames = self.image_normalization(x_frames)

        # Apply random transformations for training
        if self.hparams.use_transformations:
            x_frames = self.random_transformer(x_frames) # Apply V/H flips
            
        x_frames = x_frames.view(batch_size, self.hparams.n_frames, self.hparams.n_channels, self.hparams.img_size[0], self.hparams.img_size[1])

        # Concatenate features across frames into a single vector
        features = []

        for i in range(self.hparams.n_frames):
            features.append(self.video_encoder(x_frames[:, i, :, :, :]))

        
        # Execute FC layers on other data and append
        if self.hparams.use_force: # Force measurements
            features.append(self.force_encoder(x_forces[:, :, :].squeeze(-1)))
        if self.hparams.use_width: # Width measurements
            features.append(self.width_encoder(x_widths[:, :, :].squeeze(-1)))
        # print(len(features))
        # Send aggregated features to the FC decoder
        features = torch.cat(features, -1)

        outputs = self.decoder(features)

        # Send to decoder with deterministic estimations
        if self.hparams.use_estimations: # TODO: why is this after the decoder?
            x_estimations = torch.clamp(x_estimations, min=self.normalization_values['min_estimate'], max=self.normalization_values['max_estimate'])
            x_estimations = self.log_normalize(x_estimations, x_max=self.normalization_values['max_estimate'], x_min=self.normalization_values['min_estimate'], use_torch=True)
            outputs = self.estimation_decoder(torch.cat([outputs, x_estimations.squeeze(-1)], -1))

        return outputs
    
    def loss(self, outputs, y):

        loss = nn.functional.mse_loss(outputs.squeeze(), y.squeeze())

        # Add regularization to loss
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.params:
            l2_reg += torch.norm(param)
        loss += 0.00005 * l2_reg
        return loss

    def training_step(self, batch, batch_idx):
        # Unpack data
        x_frames, x_forces, x_widths, x_estimations, y, _ = batch
        y = y.squeeze()
        #print(x_frames.size())
        #print(x_forces.size())
        #print(x_widths.size())
        #print(x_estimations.size())
        #print(y.size())
        batch_size = x_frames.size(0)
        # Forward model
        outputs = self.forward(x_frames, x_forces, x_widths, x_estimations, batch_size=batch_size)
        outputs = outputs.squeeze()
        #print(outputs.size())
           
        loss = self.loss(outputs, y)
            
        # Calculate performance metrics
        abs_log_diff = torch.abs(torch.log10(self.log_unnormalize(outputs.cpu())) - torch.log10(self.log_unnormalize(y.cpu()))).detach()
        abs_diff = torch.abs(self.log_unnormalize(outputs.cpu()) - self.log_unnormalize(y.cpu())).detach()
        abs_acc = torch.clip((self.log_unnormalize(y.cpu()).detach() - abs_diff), min=0) / self.log_unnormalize(y.cpu()).detach()
        stats = {
            'train_loss': loss,
            'train_absolute_diff': abs_diff.mean(),
            'train_absolute_diff_min': abs_diff.min(), 
            'train_absolute_diff_max': abs_diff.max(),
            'train_absolute_acc': abs_acc.mean(),
            'train_log_acc': (abs_log_diff <= 1).sum()/batch_size, # How often the prediction is within 10x of the true value
            'train_avg_log_diff': abs_log_diff.mean(),
            'train_n_mse': nn.functional.mse_loss(outputs.squeeze(), y.squeeze())
        }
        self.log_dict(stats, batch_size=batch_size)
        return loss


    def validation_step(self, batch, batch_idx):
               # Unpack data
        x_frames, x_forces, x_widths, x_estimations, y, _ = batch
        y = y.squeeze()
        batch_size = x_frames.size(0)
        # Forward model
        outputs = self.forward(x_frames, x_forces, x_widths, x_estimations, batch_size=batch_size)
        outputs = outputs.squeeze()
           
        loss = self.loss(outputs, y)

        # Calculate performance metrics
        abs_log_diff = torch.abs(torch.log10(self.log_unnormalize(outputs.cpu())) - torch.log10(self.log_unnormalize(y.cpu()))).detach()
        abs_diff = torch.abs(self.log_unnormalize(outputs.cpu()) - self.log_unnormalize(y.cpu())).detach()
        abs_acc = torch.clip((self.log_unnormalize(y.cpu()).detach() - abs_diff), min=0) / self.log_unnormalize(y.cpu()).detach()
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
        # Unpack data
        x_frames, x_forces, x_widths, x_estimations, y, _ = batch
        y = y.squeeze()
        batch_size = x_frames.size(0)
        # Forward model
        outputs = self.forward(x_frames, x_forces, x_widths, x_estimations, batch_size=batch_size)
        outputs = outputs.squeeze()
           
        loss = self.loss(outputs, y)
            
        # Calculate performance metrics
        abs_log_diff = torch.abs(torch.log10(self.log_unnormalize(outputs.cpu())) - torch.log10(self.log_unnormalize(y.cpu()))).detach()
        abs_diff = torch.abs(self.log_unnormalize(outputs.cpu()) - self.log_unnormalize(y.cpu())).detach()
        abs_acc = torch.clip((self.log_unnormalize(y.cpu()).detach() - abs_diff), min=0) / self.log_unnormalize(y.cpu()).detach()
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params, lr=self.hparams.learning_rate)
        if self.hparams.gamma is not None:
            scheduler  = lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            return optimizer