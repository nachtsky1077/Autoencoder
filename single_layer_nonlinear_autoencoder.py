import os
from auto_encoder import AutoEncoder
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image
from utils import to_img_sigmoid

batch_size = 128
num_epochs = 50
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor()
]
)

# load MNIST dataset
dataset = MNIST(root='./datasets/mnist/', 
                download=True,
                transform=img_transform)

dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True)

# create model
slnae = AutoEncoder([28*28, 128])
criterion = nn.MSELoss()
optimizer = optim.Adam(slnae.parameters(), 
                       lr=learning_rate, 
                       weight_decay=1e-5)

# training
for epoch in tqdm(range(num_epochs)):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)

        # forward
        output = slnae(img)
        loss = criterion(output, img)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch {}/{}, loss:{:.4f}'.format(epoch+1, num_epochs, loss.data))
    if (epoch+1) % 10 == 0:
        # save output
        pic = to_img_sigmoid(output.data)
        save_image(pic, './outputs/mnist/single_layer_nonlinear_autoencoder/image_epoch_{}.png'.format(epoch))
