import os
from pca import pca, compress, generate_data_matrix
import numpy as np
from utils import sigmoid, relu

def nonlinear_pca(raw_data_mat, dimensions=[20, 2]):
    # do pca to reduce dimension to dim[0]
    v1 = pca(raw_data_mat, dimensions[0])
    data_mat_c1 = compress(raw_data_mat, v=v1)

    # apply nonlinear activation to the 
    data_mat_a1 = sigmoid(data_mat_c1)

    # do a second layer pca to reduce dimension to dim[1] (usually dim[1]=2)
    v2 = pca(data_mat_a1, dimensions[1])
    data_mat_c2 = compress(data_mat_a1, v=v2)

    return data_mat_c2, v1, v2


if __name__ == '__main__':

    pass

    