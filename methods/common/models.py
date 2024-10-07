import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FFModel(nn.Module):
    def __init__(self,  input_size=3, output_size=3, layer_sizes=[32]):
        """
        FeedForward NN
        :param input_size (int): Number of input features.
        :param output_size (int): Number of output features
        :param layer_sizes (list of int): List containing the number of units for each hidden layer.
        """
        super(FFModel, self).__init__()
        self.layers = nn.ModuleList()
        current_size = input_size
        for layer_size in layer_sizes:
            self.layers.append(nn.Linear(current_size, layer_size))
            current_size = layer_size
        self.output_layer = nn.Linear(current_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
