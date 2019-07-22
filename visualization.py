import matplotlib.pyplot as plt
import argparse
from simple_autoencoder import autoencoder, get_model_by_name

markers = ['.', 'o', 'v', '^', '1', 's', 'h', 'x', 's']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# visualize a list of 2-dimensional data with label
def visualize(data, label, savefig=False):
    fig, ax = plt.subplots()
    categories = dict()
    for i, one_label in enumerate(label):
        if not one_label in categories:
            categories[one_label] = ([], [])
        categories[one_label][0].append(data[i][0])
        categories[one_label][1].append(data[i][1])
    for i, one_label in enumerate(categories):
        curr_symbol = (markers[int(one_label) // len(markers)], colors[int(one_label) % len(colors)])
        points = categories[one_label]
        ax.scatter(x=points[0], y=points[1], marker=curr_symbol[0], c=[curr_symbol[1]] * len(points[0]))

    if not savefig:
        plt.show(fig)
    else:
        pass
