import matplotlib.pyplot as plt
import argparse
from visualization import visualize
from simple_autoencoder import autoencoder, get_model_by_name
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-n', '--num_examples', required=True)
    args = vars(parser.parse_args())

    model_name = args.get('model')
    num_examples = int(args.get('num_examples'))

    model = get_model_by_name(model_name=model_name)

    batch_size = 1

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MNIST(root='./datasets/mnist/', transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    labels = []
    X = []
    # encode images to form 2-dim data
    for i, data in enumerate(dataloader):
        img, label = data
        img = img.view(img.size(0), -1)
        two_dim_encode = model.encode(img)
        X.append(two_dim_encode.detach().numpy().reshape(2,))
        labels.append(label)
        if i >= num_examples:
            break
    visualize(data=X, label=labels)
