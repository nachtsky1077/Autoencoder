from auto_encoder import LinearAutoEncoder
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import nn, optim

batch_size = 128
num_epochs = 100
learning_rate = 1e-3

# load MNIST dataset
dataset = MNIST(root='./datasets/mnist/', 
                download=True)

dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True)

for data in dataloader:
    print(data)


exit(0)

# create model
sllae = LinearAutoEncoder([28*28, 128])
criterion = nn.MSELoss()
optimizer = optim.SGD(sllae.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        
