import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.colors import colorConverter

__license__ = 'MIT'
__author__ = 'clintval'

def darken_rgb(rgb, p):
    """
    Will darken an rgb value by p percent
    """
    assert 0 <= p <= 1, "Proportion must be [0, 1]"
    return [int(x * (1 - p)) for x in rgb]


def lighten_rgb(rgb, p):
    """
    Will lighten an rgb value by p percent
    """
    assert 0 <= p <= 1, "Proportion must be [0, 1]"
    return [int((255 - x) * p + x) for x in rgb]


def is_luminous(rgb):
    new_color = []

    for c in rgb:
        if c <= 0.03928:
            new_color.append(c / 12.92)
        else:
            new_color.append(((c + 0.055) / 1.055) ** 2.4)
    L = sum([x * y for x, y in zip([0.2126, 0.7152, 0.0722], new_color)])

    return True if L < 0.179 else False


def kmeans_plot(X, y, cluster_centers, ax=None):
    import matplotlib.patheffects as path_effects
    from sklearn.metrics.pairwise import pairwise_distances_argmin_min

    if ax is None:
        ax = plt.gca()

    cmap = cm.get_cmap("Spectral")
    colors = cmap(y.astype(float) / len(cluster_centers))
    ax.scatter(*list(zip(*X)), lw=0, c=colors, s=30)

    offset = max(list(zip(*cluster_centers))[0]) * 0.2

    for i, cluster in enumerate(cluster_centers):
        index, _ = pairwise_distances_argmin_min(cluster.reshape(1, -1), Y=X)
        cluster_color = colorConverter.to_rgb(colors[index[0]])

        if is_luminous(cluster_color) is False:
            cluster_color = darken_rgb(cluster_color, 0.35)

        label = ax.text(x=cluster[0] + offset,
                        y=cluster[1],
                        s='{:d}'.format(i + 1),
                        color=cluster_color)
        label.set_path_effects([path_effects.Stroke(lw=2, foreground='white'),
                                path_effects.Normal()])

    limit = max(*ax.get_xlim(), *ax.get_xlim())

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)

    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")
    return ax


def silhouette_plot(X, y, n_clusters, ax=None):
    from sklearn.metrics import silhouette_samples, silhouette_score

    if ax is None:
        ax = plt.gca()

    # Compute the silhouette scores for each sample
    silhouette_avg = silhouette_score(X, y)
    sample_silhouette_values = silhouette_samples(X, y)

    y_lower = padding = 2
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        
        cmap = cm.get_cmap("Spectral")
        colors = cmap(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0,
                         ith_cluster_silhouette_values,
                         facecolor=color,
                         edgecolor=color,
                         alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + padding

    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax.axvline(x=silhouette_avg, c='r', alpha=0.8, lw=0.8, ls='-')
    ax.annotate('Average',
                xytext=(silhouette_avg, y_lower * 1.025),
                xy=(0, 0),
                ha='center',
                alpha=0.8,
                c='r')

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylim(0, y_upper + 1)
    ax.set_xlim(-0.075, 1.0)
    return ax