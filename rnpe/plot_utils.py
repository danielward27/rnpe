import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib.lines import Line2D


# TODO document
def pairplot(
    arrays,
    true,
    array_point_size=0.1,
    true_point_size=5,
    facet_size=5,
    names=None, # arrays and true names for legend
    col_names=None,  # summary statistic names
    match_size=True,
    colors = None,
    trim_quantile = 0.0005,  # To remove big outliers that interfere with plotting
    dpi = 100
    ):
    if colors is None:
        colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
        colors.remove("tab:green")  # save for true    
        colors = colors[:len(arrays)]
        colors.append("tab:green")

    dim = len(true)
    
    if col_names is None:
        col_names = ["" for i in range(dim)]
    arrays = [np.array(a) for a in arrays]
    true = np.array(true)

    if match_size:
        min_size = min([a.shape[0] for a in arrays])
        arrays = [a[:min_size] for a in arrays]

    if trim_quantile:
        stacked = np.concatenate(arrays)
        l = np.quantile(stacked, trim_quantile, axis=0),
        u = np.quantile(stacked, 1-trim_quantile, axis=0)
        arrays_trimmed = [
            _trim(a, l, u) for a in arrays
        ]
        arrays = arrays_trimmed
        assert len(arrays)

    fig, axs = plt.subplots(dim, dim, dpi=dpi)
    
    
    for j in range(dim):
        for i in range(dim):
            if i < j:
                axs[i, j].axis("off")
            else:
                for a, color in zip(arrays, colors[:-1]):
                    if i==j:
                        sns.kdeplot(a[:, i], ax = axs[i, j], color=color)
                        axs[i, j].axvline(true[i], color=colors[-1])
                        axs[i, j].set_ylabel("")
                    else:
                        axs[i, j].scatter(a[:, j], a[:, i], s=array_point_size, alpha=0.1, color=color)
                        axs[i, j].scatter(true[j], true[i], s=true_point_size, color=colors[-1])
            
            if i >= j:
                if i==(dim-1):
                    axs[i, j].set_xlabel(col_names[j])
                if j==0:
                    axs[i, j].set_ylabel(col_names[i])

            axs[i,j].xaxis.set_ticklabels([])
            axs[i,j].yaxis.set_ticklabels([])
        size = dim**0.5  # Seems reasonable default
        fig.set_size_inches(facet_size*size, facet_size*size)

    if names is not None:
        legend_elements = _get_manual_legend(names, colors)
        plt.figlegend(legend_elements, names, loc = "lower center", ncol=len(names))

def _trim(a, l, u):
    a = a[(a > l).all(axis=1)]
    a = a[(a < u).all(axis=1)]
    return a

def _get_manual_legend(labels, colors):
    legend_elements = [  # Manual legend to avoid small points in legend
        Line2D([0], [0], marker='o', color='w', label=label,
        markerfacecolor=color, markersize=8) for label, color in zip(labels, colors)
        ]
    return legend_elements

    
