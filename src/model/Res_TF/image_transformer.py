import torch.nn as nn
import torchvision.models as models
import torch

from model.utils import conv2D_output_size, conv3D_output_size


class ImageTransformerWithPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, num_layers=5, dropout=0.1, output_dim=512):
        super(ImageTransformerWithPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Update ResNet feature extractor to handle custom input size
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Calculate the output size of the feature extractor given input size (250, 350)
        self.feature_size = self._calculate_feature_size((250, 350))
        
        # Projecting ResNet features to Transformer dimension
        self.feature_projection = nn.Linear(512, embedding_dim)
        
        # Positional encoding for three positions
        self.positional_encoding = self._generate_positional_encoding(3, embedding_dim)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layer to map to the final embedding dimension
        self.fc = nn.Linear(embedding_dim, output_dim)  # Adjust according to the desired output dimension
        self.dropout = nn.Dropout(p=dropout)
    
    def _calculate_feature_size(self, input_size):
        # Define the layers' parameters
        layers = [
            {'kernel_size': (7, 7), 'stride': (2, 2), 'padding': (3, 3)},
            {'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (1, 1)},
            {'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (1, 1)},
            {'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (1, 1)},
            {'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (1, 1)}
        ]
        size = input_size
        for conv in layers:
            size = conv2D_output_size(size, conv['padding'], conv['kernel_size'], conv['stride'])
        return size
    
    def _generate_positional_encoding(self, num_positions, d_model):
        pe = torch.zeros(num_positions, d_model)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, images):
        batch_size = images.size(0)
        
        # Extract features from each image
        features = []
        for i in range(images.size(1)):
            # Extract features for image i
            feature = self.feature_extractor(images[:, i])
            feature = feature.mean([2, 3])  # Average pooling
            features.append(feature)
        
        # Stack features and project to Transformer dimension
        features = torch.stack(features, dim=1)  # Shape: (batch_size, 3, 512)
        
        features = self.feature_projection(features)  # Shape: (batch_size, 3, embedding_dim)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding.unsqueeze(0).to(features.device)  # Shape: (1, 3, embedding_dim)
        features += pos_encoding
        
        # Prepare for Transformer input
        features = features.permute(1, 0, 2)  # Shape: (3, batch_size, embedding_dim)
        
        # Transformer forward pass
        transformer_output = self.transformer_encoder(features)  # Shape: (3, batch_size, embedding_dim)
        
        # Use the output corresponding to the first image (or apply pooling if needed)
        output = transformer_output.mean(dim=0)  # Shape: (batch_size, embedding_dim)
        
        # Fully connected layer to get the final embedding
        output = self.dropout(output)
        embedding = self.fc(output)  # Shape: (batch_size, 1024)
        
        
        return embedding