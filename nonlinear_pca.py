import os
from pca import pca, compress, generate_data_matrix
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from utils import sigmoid, relu
from visualization import visualize
import argparse
from sklearn.preprocessing import StandardScaler

def nonlinear_pca(raw_data_mat, dimensions=[20, 2], nonlinear_func=sigmoid):
    # do pca to reduce dimension to dim[0]
    v1 = pca(raw_data_mat, dimensions[0])
    data_mat_c1 = compress(raw_data_mat, v=v1)

    # apply nonlinear activation to the 
    data_mat_a1 = nonlinear_func(data_mat_c1)

    # do a second layer pca to reduce dimension to dim[1] (usually dim[1]=2)
    v2 = pca(data_mat_a1, dimensions[1])
    data_mat_c2 = compress(data_mat_a1, v=v2)
    data_mat_a2 = sigmoid(data_mat_c2)

    return data_mat_a2, v1, v2


if __name__ == '__main__':
    
    # command line arguments parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_examples', required=True)
    parser.add_argument('-k', '--dimension', required=True)
    parser.add_argument('-l', '--show_labels')
    
    args = vars(parser.parse_args())
    batch_size = 1
    show_labels = args.get('show_labels', None)
    try:
        show_labels = [int(one_label) for one_label in  show_labels.split(',')]
    except Exception as e:
        print('Illegal argument show_labels. Set to None.')
        show_labels = None
    k = int(args.get('dimension'))
    num_images = int(args.get('num_examples'))
    
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

    datamat_c, v1, v2 = nonlinear_pca(raw_datamat_normalized, dimensions=[k, 2], nonlinear_func=sigmoid)

    visualize(datamat_c, labels, show_labels=show_labels)


    