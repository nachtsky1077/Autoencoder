


def tensor_to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def to_img_sigmoid(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x