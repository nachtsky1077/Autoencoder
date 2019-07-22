import time
import os

def tensor_to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def to_img_sigmoid(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x


def exec_time(func):
    def inner(*args, **kw):
        ts = time.time()
        func(*args, **kw)
        te = time.time()
        print('Total execution time: {:10.4f} secs'.format(te-ts))
    return inner

def get_model_by_name(model_name, base_path='outputs/mnist/models/'):
    model_fu