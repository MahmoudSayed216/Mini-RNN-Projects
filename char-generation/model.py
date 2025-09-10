import torch
import torch.nn as nn


class RNNSanityCheck(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super(RNNSanityCheck, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.RNN = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden):
        x, hidden = self.RNN(x, hidden)
        x = self.classifier(x)
        return x, hidden
    

    def no_history_state_vector(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)