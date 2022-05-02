import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib.lines import Line2D


def pairplot(
    arrays: dict,
    true: np.ndarray,
    col_names: list = None,  # summary statistic names
    array_point_size: float = 0.4,
    true_point_size: float = 20,
    facet_size: float = 5,
    match_size: bool = True,
    colors=None,
    trim_quantile=0.0005,  # remove big outliers that may interfere with plotting
    dpi=100,
    show_x_axis=False,
    legend_y_adjust=0.02,
    alpha=0.2,
    true_name="true",
    exclude_diag=False,
):
    """Plot a pairplot, between the different columns of each array, along with 
    an additional true/reference point. Note this function does remove extreme
    outliers, and matches the sizes between the arrays by default by subsampling.

    Args:
        arrays (list): dictionary with names as keys and arrays as values, with arrays matching dimension on axis 1.
        true (np.ndarray): True/reference point.
        col_names (list): list of names corresponding the columns of each array (e.g. summary statistics).
        array_point_size (float, optional): Point size. Defaults to 0.1.
        true_point_size (float, optional): Point size. Defaults to 5.
        facet_size (float, optional): Facet size. Defaults to 5.
        colors (list, optional): Colours, list of length len(arrays) + 1 (last colour for true point).
        trim_quantile (float, optional): Trim outliers. Defaults to 0.0005.
        show_x_axis (bool, optional): S. Defaults to False.
        legend_y_adjust (float, optional): Nudge legend in y axis. Defaults to 0.02.
        alpha (float, optional): Opacity of array points. Defaults to 0.2.
        exclude_diag: (bool): Option to exlude density plots along diagonal.
    """
    if colors is None:
        colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
        colors.remove("tab:green")  # save for true
        colors = colors[: len(arrays)]
        colors.append("tab:green")

    array_names = list(arrays.keys())
    array_names.append(true_name)
    arrays = [np.array(a) for a in arrays.values()]
    true = np.array(true)
    dim = arrays[0].shape[1]

    if col_names is None:
        col_names = ["" for i in range(dim)]

    if match_size:
        ns = [a.shape[0] for a in arrays]
        min_size = min(ns)
        rng = np.random.default_rng()
        idxs = [rng.choice(np.arange(n), min_size) for n in ns]
        arrays = [a[i] for (i, a) in zip(idxs, arrays)]

    if trim_quantile:
        stacked = np.concatenate(arrays)
        l = np.quantile(stacked, trim_quantile, axis=0)
        u = np.quantile(stacked, 1 - trim_quantile, axis=0)
        arrays_trimmed = [_trim(a, l, u) for a in arrays]
        arrays = arrays_trimmed
        assert len(arrays)

    fig, axs = plt.subplots(dim, dim, dpi=dpi)

    for j in range(dim):
        for i in range(dim):
            ax = axs if dim == 1 else axs[i, j]

            if i < j:
                ax.axis("off")
            else:
                for a, color in zip(arrays, colors[:-1]):
                    if i == j:
                        sns.kdeplot(a[:, i], ax=ax, color=color)
                        ax.set_ylabel("")
                        ax.axvline(true[i], color=colors[-1])
                    else:
                        ax.scatter(
                            a[:, j],
                            a[:, i],
                            s=array_point_size,
                            alpha=alpha,
                            color=color,
                            edgecolors="none",
                            rasterized=True,
                        )
                        ax.scatter(
                            true[j], true[i], s=true_point_size, color=colors[-1]
                        )

            if i >= j:
                if i == (dim - 1):
                    ax.set_xlabel(col_names[j])
                if j == 0:
                    ax.set_ylabel(col_names[i])

            if show_x_axis is False:
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

        size = dim ** 0.5  # Seems reasonable default
        fig.set_size_inches(facet_size * size, facet_size * size)

    legend_elements = _get_manual_legend(array_names, colors)
    plt.figlegend(
        legend_elements,
        array_names,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y_adjust),
        ncol=len(array_names),
    )
    return fig


def _trim(a, l, u):
    a = a[(a > l).all(axis=1)]
    a = a[(a < u).all(axis=1)]
    return a


def _get_manual_legend(labels, colors):
    legend_elements = [  # Manual legend to avoid small points in legend
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=8,
        )
        for label, color in zip(labels, colors)
    ]
    return legend_elements
