import os
from auto_encoder import LinearAutoEncoder
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image
from utils import tensor_to_img

batch_size = 128
num_epochs = 20
learning_rate = 1e-3

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

# create model
sllae = LinearAutoEncoder([28*28, 128])
criterion = nn.MSELoss()
optimizer = optim.SGD(sllae.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)

        # forward
        output = sllae(img)
        loss = criterion(output, img)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch {}/{}, loss:{:.4f}'.format(epoch+1, num_epochs, loss.data))


# save output
pic = tensor_to_img(output.data)
save_image(pic, './outputs/mnist/single_layer_linear_autoencoder/image_epoch_{}.png'.format(epoch))