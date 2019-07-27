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

def nonlinear_pca(raw_data_mat, dimensions=[20, 10, 2], nonlinear_func=sigmoid):
    input_data_mat = raw_data_mat
    for dim in dimensions:
        # do pca to reduce dimension
        v = pca(input_data_mat, dim)
        data_mat_c = compress(input_data_mat, v=v)
        data_mat_a = nonlinear_func(data_mat_c)
        input_data_mat = data_mat_a
    return data_mat_a


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
        show_labels = [int(one_label) for one_label in show_labels.split(',')]
    except Exception as e:
        print('Illegal argument show_labels. Set to None.')
        show_labels = None
    k = args.get('dimension')
    k = [int(one_k) for one_k in k.split(',')]
    k.append(2)
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

    datamat_c = nonlinear_pca(raw_datamat_normalized, dimensions=k, nonlinear_func=sigmoid)

    visualize(datamat_c, labels, show_labels=show_labels)


    