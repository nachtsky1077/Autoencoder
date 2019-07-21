import torch
from torch import nn



class LinearAutoEncoder(nn.Module):

    def __init__(self, layer_specs):
        super(LinearAutoEncoder, self).__init__()
        
        # FIXME: encoder and decoder ought to be not only arch-symmetric but weight-symmetric?
        encode_layers = []
        for i in range(1, len(layer_specs)):
            encode_layers.append(nn.Linear(layer_specs[i-1], layer_specs[i]))
        self.encoder = nn.Sequential(*encode_layers)

        decode_layers = []
        for i in range(len(layer_specs)-2, -1, -1):
            decode_layers.append(nn.Linear(layer_specs[i+1], layer_specs[i]))
        self.decoder = nn.Sequential(*decode_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder(nn.Module):

    def __init__(self, layer_specs, nonlinear_func=nn.Sigmoid):
        super(AutoEncoder, self).__init__()

        encode_layers = []
        for i in range(1, len(layer_specs)):
            encode_layers.append(nn.Linear(layer_specs[i-1], layer_specs[i]))
            # add non-linear activation
            encode_layers.append(nonlinear_func())
        self.encoder = nn.Sequential(*encode_layers)

        decode_layers = []
        for i in range(len(layer_specs)-2, -1, -1):
            decode_layers.append(nn.Linear(layer_specs[i+1], layer_specs[i]))
            # add non_linear activation
            decode_layers.append(nonlinear_func())
        self.decoder = nn.Sequential(*decode_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
    

