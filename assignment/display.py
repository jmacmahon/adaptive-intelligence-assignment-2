import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from itertools import combinations


def get_3d_tunings_figures(tunings, labels=None):
    """Compute 3-D surface graphs of parametrised performance data."""
    parameters = tunings.shape[1] - 1
    combinations_ = list(combinations(range(parameters), r=2))

    figures = {}
    for x_index, y_index in combinations_:
        x_values = np.unique(tunings[:, x_index])
        y_values = np.unique(tunings[:, y_index])
        grid = np.meshgrid(x_values, y_values)
        z_values_array = []
        for x_value, y_value in zip(grid[0].reshape(-1), grid[1].reshape(-1)):
            selector = np.all([tunings[:, x_index] == x_value,
                               tunings[:, y_index] == y_value], axis=0)
            z_value = np.mean(tunings[selector, parameters])
            z_values_array.append(z_value)
        z_values = np.array(z_values_array).reshape(grid[0].shape)

        fig = plt.figure()
        figures[(x_index, y_index)] = fig
        ax = fig.add_subplot(111, projection='3d')
        if labels is not None:
            ax.set_xlabel(labels[x_index])
            ax.set_ylabel(labels[y_index])
        ax.plot_surface(*grid, z_values)
    return figures
