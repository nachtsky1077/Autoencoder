import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from utils import to_img_sigmoid
'''
Load N images from the dataset, vectorize the images and then stack the vectorized
images to form a matrix.
'''
def generate_data_matrix(dataloader, num_images, image_size=(28, 28)):
    data_mat = np.zeros((num_images, np.prod(image_size, axis=None)))
    for i, data in enumerate(dataloader):
        if i == num_images:
            break
        img, _ = data
        img = img.view(-1).numpy()
        data_mat[i] = img.reshape(1, img.size)
    return data_mat

def pca(raw_data_mat, k=10):
    '''
    raw_data_mat: a numpy array wish shape N x D (N: number of data samples, D: number of dimensions)
    '''
    # FIXME: the raw_data_mat is not zero-mean
    covmat = np.matmul(np.transpose(raw_data_mat), raw_data_mat) / (raw_data_mat.shape[0] - 1)
    
    w, v = np.linalg.eig(covmat)
    # ignore the small imginary part of the eigen vectors as it comes from numeric error
    return np.real(v[:, :k])

def pca_est(raw_data_mat, v):
    # FIXME: np.matmul(raw_data_mat, V) is sufficient?
    return np.matmul(np.matmul(raw_data_mat, v), np.transpose(v))


# implment 2 layers of pca
batch_size = 1
num_images = 20000

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
                        shuffle=False)

raw_datamat = generate_data_matrix(dataloader, num_images)

# save some original images
grid_size = 3 * 3
original_images = torch.from_numpy(raw_datamat[:grid_size])
original_images = original_images.view(grid_size, 1, 28, 28)
save_image(original_images, './outputs/pca/n={}_original_images.png'.format(num_images), nrow=3)
k = 50

v = pca(raw_datamat, k)
est_datamat = pca_est(raw_datamat, v)
estimated_images = torch.from_numpy(est_datamat[:grid_size])
estimated_images = estimated_images.view(grid_size, 1, 28, 28)
save_image(estimated_images, './outputs/pca/n={}_pca_estimation_k={}.png'.format(num_images, k), nrow=3)