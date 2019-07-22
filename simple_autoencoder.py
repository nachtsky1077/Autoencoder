import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import pickle
import getopt


if not os.path.exists('outputs/mnist/mlp_img'):
    os.mkdir('outputs/mnist/mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 5
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST(root='./datasets/mnist/', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 2))
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

try:  
    # TODO: load model and training epoches from state_dict
    raise 1
except:
    # create a new model
    model = autoencoder()#.cuda()
    base_epoch = 0

losses = dict()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        #img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data))
    # ================store loss====================
    losses[epoch + base_epoch] = loss.data
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, 'outputs/mnist/mlp_img/image_{}.png'.format(epoch))
torch.save(model.state_dict(), 'outputs/mnist/models/mnist_sim_autoencoder_epoch{}.pth'.format(epoch + base_epoch))
with open('outputs/mnist/models/mnist_sim_autoencoder_training_loss_epoch_{}_{}.pkl'.format(base_epoch, base_epoch + epoch - 1), 'wb') as f:
    pickle.dump(losses, f)

