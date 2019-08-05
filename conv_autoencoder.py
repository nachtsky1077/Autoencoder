import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # encoder layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, 2, padding=0)
        self.conv4 = nn.Conv2d(4, 2, 3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        self.t_conv1 = nn.ConvTranspose2d(2, 4, 3, stride=3)
        self.t_conv2 = nn.ConvTranspose2d(4, 4, 2, stride=2, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(16, 1, 2, stride=2)
    
    def encode(self, x):
        # encode
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        return x
    
    def forward(self, x):
        # encode
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        
        # decode
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = torch.sigmoid(self.t_conv4(x))

        return x
