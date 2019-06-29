import torch
from torch import nn



class LinearAutoEncoder(nn.Module):

    def __init__(self, layer_specs):
        super(LinearAutoEncoder, self).__init__()
        
        encode_layers = []
        for i, curr_layer_feats in enumerate(layer_specs[1:]):
            encode_layers.append(nn.Linear(layer_specs[i-1], curr_layer_feats))
        self.encoder = nn.Sequential(*encode_layers)

        decode_layers = []
        for i, curr_layer_feats in enumerate(layer_specs[-2::-1]):
            decode_layers.append(nn.Linear(layer_specs[i+1], curr_layer_feats))
        self.decoder = nn.Sequential(*decode_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
    

