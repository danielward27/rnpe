import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib.lines import Line2D


# TODO document
def pairplot(
    arrays,
    true=None,
    array_point_size=0.1,
    true_point_size=5,
    facet_size=5,
    names=None,  # arrays and true names for legend
    col_names=None,  # summary statistic names
    match_size=True,
    colors=None,
    trim_quantile=0.0005,  # remove big outliers that may interfere with plotting
    dpi=100,
    show_x_axis=False,
    legend_y_adjust=0.02,
    alpha=0.2,
):
    if colors is None:
        colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
        colors.remove("tab:green")  # save for true
        colors = colors[: len(arrays)]
        colors.append("tab:green")

    dim = arrays[0].shape[1]

    if col_names is None:
        col_names = ["" for i in range(dim)]
    arrays = [np.array(a) for a in arrays]
    if true is not None:
        true = np.array(true)

    if match_size:
        min_size = min([a.shape[0] for a in arrays])
        arrays = [a[:min_size] for a in arrays]

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
                        if true is not None:
                            ax.axvline(true[i], color=colors[-1])
                    else:
                        ax.scatter(
                            a[:, j],
                            a[:, i],
                            s=array_point_size,
                            alpha=alpha,
                            color=color,
                        )
                        if true is not None:
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

    if names is not None:
        legend_elements = _get_manual_legend(names, colors)
        plt.figlegend(
            legend_elements,
            names,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y_adjust),
            ncol=len(names),
        )


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
