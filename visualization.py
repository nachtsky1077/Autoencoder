import matplotlib.pyplot as plt
import argparse
from simple_autoencoder import autoencoder, get_model_by_name

markers = ['.', 'v', '^', '1', 's', 'h', 'x', 's']
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# visualize a list of 2-dimensional data with label
def visualize(data, label, show_labels=None, num_each_cat=500, savefig=False):
    fig, ax = plt.subplots()
    categories = dict()
    for i, one_label in enumerate(label):
        if not one_label.item() in categories:
            categories[one_label.item()] = ([], [])
        categories[one_label.item()][0].append(data[i][0])
        categories[one_label.item()][1].append(data[i][1])
    i = 0
    
    for one_label in categories:
        if show_labels is not None and not one_label in show_labels:
            continue
        curr_symbol = (markers[int(one_label) // len(colors)], colors[int(one_label) % len(colors)])
        points = categories[one_label]
        ax.scatter(x=points[0][:num_each_cat], y=points[1][:num_each_cat], label=one_label, marker=curr_symbol[0], c=[curr_symbol[1]] * min(num_each_cat, len(points[0])))
    ax.legend()
    if not savefig:
        plt.show(fig)
    else:
        pass
