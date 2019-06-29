import os
from auto_encoder import LinearAutoEncoder
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import nn, optim
from torchvision import transforms

batch_size = 128
num_epochs = 100
learning_rate = 1e-3

def pil_to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)

# load MNIST dataset
dataset = MNIST(root='./datasets/mnist/', 
                download=True,
                transform=img_transform)

dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True)

for data in dataloader:
    img_batch, labels = data
    print(img_batch.shape)


exit(0)

# create model
sllae = LinearAutoEncoder([28*28, 128])
criterion = nn.MSELoss()
optimizer = optim.SGD(sllae.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        
