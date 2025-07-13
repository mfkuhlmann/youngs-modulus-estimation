import torch
import torch.nn as nn

from model.VGG_LSTM.lstm_layer import SequentialLSTM

class TripleLSTMModel(nn.Module):
    def __init__(self, input_dim, scalar1_dim, scalar2_dim, scalar3_dim, scalar4_dim, hidden_dim, dropout_rate):
        super(TripleLSTMModel, self).__init__()

        self.lstm_seq = SequentialLSTM(input_dim, scalar1_dim, scalar2_dim, scalar3_dim, scalar4_dim, hidden_dim, dropout_rate)

        self.weights = nn.Parameter(torch.ones(3))
        

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, images=None, forces=None, widths=None, scalars3=None, scalars4=None):

        outputs = self.lstm_seq(images, forces, widths, scalars3, scalars4)
        

        weights = nn.functional.softmax(self.weights, dim=0)  
        output = sum(w * y for w, y in zip(weights, outputs))
        #utput = F.elu(output)
        output = self.sigmoid(output)
        return output