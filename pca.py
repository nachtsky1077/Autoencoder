import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from utils import to_img_sigmoid
import argparse
from visualization import visualize
from sklearn.preprocessing import StandardScaler

'''
Load N images from the dataset, vectorize the images and then stack the vectorized
images to form a matrix.
'''
def generate_data_matrix(dataloader, num_images, image_size=(28, 28)):
    data_mat = np.zeros((num_images, np.prod(image_size, axis=None)))
    labels = []
    for i, data in enumerate(dataloader):
        if i == num_images:
            break
        img, label = data
        img = img.view(-1).numpy()
        data_mat[i] = img.reshape(1, img.size)
        labels.append(label)
    return data_mat, labels

def pca(raw_data_mat, k=10):
    '''
    raw_data_mat: a numpy array with shape N x D (N: number of data samples, D: number of dimensions)
    '''
    covmat = np.matmul(np.transpose(raw_data_mat), raw_data_mat) / (raw_data_mat.shape[0] - 1)

    w, v = np.linalg.eig(covmat)
    # ignore the small imginary part of the eigen vectors as it comes from numeric error
    return np.real(v[:, :k])

def pca_est(raw_data_mat, v):
    # FIXME: np.matmul(raw_data_mat, V) is sufficient?
    return np.matmul(np.matmul(raw_data_mat, v), np.transpose(v))

def compress(raw_data_mat, v):
    return np.matmul(raw_data_mat, v)

if __name__ == '__main__':
    
    # command line arguments parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_examples', required=True)
    parser.add_argument('-k', '--dimension', required=True)
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-l', '--show_labels')

    args = vars(parser.parse_args())
    batch_size = 1
    show_labels = args.get('show_labels', None)
    try:
        show_labels = [int(one_label) for one_label in  show_labels.split(',')]
    except Exception as e:
        print('Illegal argument show_labels. Set to None')
        show_labels = None
        
    k = int(args.get('dimension'))
    num_images = int(args.get('num_examples'))
    save_fig = args.get('v')
    
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

    raw_datamat, labels = generate_data_matrix(dataloader, num_images)
    
    scaler = StandardScaler(with_std=False)
    scaler.fit(raw_datamat)

    # removing mean from raw_data
    raw_datamat_normalized = scaler.transform(raw_datamat)

    grid_size = 3 * 3
    original_images = torch.from_numpy(raw_datamat_normalized[:grid_size])
    original_images = original_images.view(grid_size, 1, 28, 28)
    v = pca(raw_datamat_normalized, k)
    est_datamat = pca_est(raw_datamat_normalized, v)
    est_datamat = scaler.inverse_transform(est_datamat)
    estimated_images = torch.from_numpy(est_datamat[:grid_size])
    estimated_images = estimated_images.view(grid_size, 1, 28, 28)
    
    # could only visualize 2-dimensional data
    if k == 2:
        visualize(compress(raw_datamat_normalized, v), labels, show_labels=show_labels)

    if save_fig:
        # save some original images
        save_image(original_images, './outputs/pca/n={}_original_images.png'.format(num_images), nrow=3)
        save_image(estimated_images, './outputs/pca/n={}_pca_estimation_k={}.png'.format(num_images, k), nrow=3)
