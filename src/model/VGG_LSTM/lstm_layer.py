import torch
import torch.nn as nn
from model.VGG_LSTM.extractor_layer import VGG16FC7Extractor

class SequentialLSTM(nn.Module):
    def __init__(self, input_dim, scalar1_dim, scalar2_dim, scalar3_dim, scalar4_dim, hidden_dim, dropout_rate):
        super(SequentialLSTM, self).__init__()
    
        self.feature_extractor = VGG16FC7Extractor()
  
        self.force_mapper = nn.Linear(1, scalar1_dim)
   
        self.width_mapper = nn.Linear(1, scalar2_dim)

        self.scalar3_mapper = nn.Linear(1, scalar3_dim)

        self.scalar4_mapper = nn.Linear(1, scalar4_dim)
        

        self.lstm_cell = nn.LSTMCell(input_dim + scalar1_dim + scalar2_dim + scalar3_dim + scalar4_dim, hidden_dim)
        
        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
  
        self.fc = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.scalar1_dim = scalar1_dim
        self.scalar2_dim = scalar2_dim
        self.scalar3_dim = scalar3_dim
        self.scalar4_dim = scalar4_dim

    def forward(self, images, forces=None, widths=None, scalars3=None, scalars4=None):
        batch_size = images[0].size(0)


        h_t = torch.zeros(batch_size, self.hidden_dim).to(images[0].device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(images[0].device)
        outputs = []
        #print(len(images))
        for i in range(len(images)):
            
            feature = self.feature_extractor(images[i])
            feature = feature.view(batch_size, -1)  
         
            if forces is not None:
                force_mapped = self.force_mapper(forces[i])
            else:
                force_mapped = torch.zeros(batch_size, self.scalar1_dim).to(images[0].device)
            
           
            if widths is not None:
                width_mapped = self.width_mapper(widths[i])
            else:
                width_mapped = torch.zeros(batch_size, self.scalar2_dim).to(images[0].device)
                
         
            if scalars3 is not None:
                scalar3_mapped = self.scalar3_mapper(scalars3[i])
            else:
                scalar3_mapped = torch.zeros(batch_size, self.scalar3_dim).to(images[0].device)
                
         
            if scalars4 is not None:
                scalar4_mapped = self.scalar4_mapper(scalars4[i])
            else:
                scalar4_mapped = torch.zeros(batch_size, self.scalar4_dim).to(images[0].device)
            
          
            concatenated_features = torch.cat((feature, force_mapped, width_mapped, scalar3_mapped, scalar4_mapped), dim=1)
            
        
            h_t, c_t = self.lstm_cell(concatenated_features, (h_t, c_t))
            
            # Apply dropout to the hidden state
            h_t = self.dropout(h_t)
            
    
            y_t = self.fc(h_t)
            outputs.append(y_t)
        
        return outputs