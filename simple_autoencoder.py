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
import argparse
import logging
from utils import exec_time

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

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

def get_model_by_name(model_name, base_path='outputs/mnist/models/'):
    # load trained model
    model_full_path = os.path.join(base_path, model_name)
    model = autoencoder()
    model.load_state_dict(torch.load(model_full_path))
    model.eval()
    return model
    
def get_model(epoch=0, base_path='outputs/mnist/models/'):
    if epoch == 0:
        # create a new model
        model = autoencoder()#.cuda()
    else:
        # load trained model
        model_full_path = os.path.join(base_path, 'mnist_sim_autoencoder_epoch{}.pth'.format(epoch))
        model = autoencoder()
        model.load_state_dict(torch.load(model_full_path))
        model.eval()
    return model

@exec_time
def train(model, dataloader, base_epoch, num_epochs=10):
    # for reconstruction loss tracking
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
    with open('outputs/mnist/models/mnist_sim_autoencoder_training_loss_epoch_{}_{}.pkl'.format(base_epoch, base_epoch + epoch), 'wb') as f:
        pickle.dump(losses, f)


if __name__ == '__main__':

    # command line arguments parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--start_epoch', required=True)
    parser.add_argument('-n', '--num_epochs')
    args = vars(parser.parse_args())
    
    base_epoch = int(args.get('start_epoch'))
    num_epochs = int(args.get('num_epochs', 10))

    if not os.path.exists('outputs/mnist/mlp_img'):
        os.mkdir('outputs/mnist/mlp_img')

    # fixed batch_size and learning rate
    batch_size = 128
    learning_rate = 1e-3

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MNIST(root='./datasets/mnist/', transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # fetch model, create one if base_epoch is 0
    model = get_model(epoch=base_epoch, base_path='outputs/mnist/models/')
    train(model=model, dataloader=dataloader, base_epoch=base_epoch, num_epochs=num_epochs)
    