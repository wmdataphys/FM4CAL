import json
import math
import h5py
import os
import awkward as ak
import webbrowser
from datetime import datetime

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import seaborn as sns
import vector
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance
import torch

vector.register_awkward()

import utils.plotting_utils as plot_utils
from utils.metrics import quantiled_kl_divergence
from utils.plotting_utils import (
    KL,
    find_max_energy_z,
    get_COG_ak,
    sum_energy_per_layer,
    sum_energy_per_radial_distance,
    write_distances_to_json,
    plot_ratios,plot_ratios_np,
    sum_energy_per_layer_unc,
    sum_energy_per_radial_distance_unc
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots   


# Base plotting code taken directly from Omnijet Alpha-c: https://github.com/uhh-pd-ml/omnijet_alpha_c

def binclip(x, bins, dropinf=False):
    binfirst_center = bins[0] + (bins[1] - bins[0]) / 2
    binlast_center = bins[-2] + (bins[-1] - bins[-2]) / 2
    if dropinf:
        print("Dropping inf")
        print("len(x) before:", len(x))
        x = x[~np.isinf(x)]
        print("len(x) after:", len(x))
    return np.clip(x, binfirst_center, binlast_center)


def get_bin_centers_and_bin_heights_from_hist(hist):
    """Return the bin centers and bin heights from a histogram.

    Parameters
    ----------
    hist : tuple
        The output of matplotlib hist.

    Returns
    -------
    bin_centers : array-like
        The bin centers.
    bin_heights : array-like
        The bin heights.
    """
    bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
    bin_heights = hist[0]
    return bin_centers, bin_heights


def plot_hist_with_ratios(
    comp_dict: dict,
    bins: np.ndarray,
    ax_upper: plt.Axes,
    ax_ratio: plt.Axes = None,
    ref_dict: dict = None,
    ratio_range: tuple = None,
    xlabel: str = None,
    logy: bool = False,
    leg_loc: str = "best",
    underoverflow: bool = True,
    leg_title: str = None,
    leg_ncols: int = 1,
    return_hist_curve: bool = False,
):
    """Plot histograms of the reference and comparison arrays, and their ratio.

    Parameters:
    ----------
    ax_upper : plt.Axes
        Axes for the upper panel.
    ax_ratio : plt.Axes
        Axes for the ratio panel.
    ref_dict : dict
        Dict with {id: {"arr": ..., "hist_kwargs": ...}, ...} of the reference array.
    comp_dict : dict
        Dict with {id: {"arr": ..., "hist_kwargs": ...}, ...} of the comparison arrays.
    bins : np.ndarray
        Bin edges for the histograms.
    ratio_range : tuple, optional
        Range of the y-axis for the ratio plot.
    xlabel : str, optional
        Label for the x-axis.
    logy : bool, optional
        Whether to plot the y-axis in log scale.
    leg_loc : str, optional
        Location of the legend.
    underoverflow : bool, optional
        Whether to include underflow and overflow bins. Default is True.
    leg_title : str, optional
        Title of the legend.
    leg_ncols : int, optional
        Number of columns in the legend. Default is 1.
    return_hist_curve : bool, optional
        Whether to return the histogram curves in a dict. Default is False.

    Returns
    -------
    hist_curve_dict : dict
        Dict with {id: (bin_centers, bin_heights), ...} of the histogram curves.
        Only returned if `return_hist_curve` is True. Both bin_centers and bin_heights
        are array-like.
    """

    legend_handles = []
    hist_curve_dict = {}

    if ref_dict is not None:
        ref_arr = list(ref_dict.values())[0]
        ref_label = list(ref_dict.keys())[0]
        kwargs_ref = dict(histtype="stepfilled", color="k", alpha=0.25, label=ref_label)

    if leg_title is not None:
        # plot empty array with alpha 0 to create a legend entry
        ax_upper.hist([], alpha=0, label=leg_title)

    kwargs_common = dict(bins=bins, density=True)
    if ref_dict is not None:
        hist_ref = ax_upper.hist(binclip(ref_arr["arr"], bins), **kwargs_common, **kwargs_ref)

    if ax_ratio is not None:
        ax_ratio.axhline(1, color="black", linestyle="--", lw=1)

    # loop over entries in comp_dict and plot them
    for i, (arr_id, arr_dict) in enumerate(comp_dict.items()):
        kwargs_comp = dict(histtype="step") | arr_dict.get("hist_kwargs", {})
        if "linestyle" in kwargs_comp:
            if kwargs_comp["linestyle"] == "dotted":
                kwargs_comp["linestyle"] = plot_utils.get_good_linestyles("densely dotted")
        hist_comp = ax_upper.hist(binclip(arr_dict["arr"], bins), **kwargs_common, **kwargs_comp)
        if return_hist_curve:
            hist_curve_dict[arr_id] = get_bin_centers_and_bin_heights_from_hist(hist_comp)
        legend_handles.append(
            Line2D(
                [],
                [],
                color=kwargs_comp.get("color", "C1"),
                lw=kwargs_comp.get("lw", 1),
                label=kwargs_comp.get("label", arr_id),
                linestyle=kwargs_comp.get("linestyle", "-"),
            )
        )
        if ax_ratio is not None:
            # calculate and plot ratio
            ratio = hist_comp[0] / hist_ref[0]
            # duplicate the first entry to avoid a gap in the plot (due to step plot)
            ratio = np.append(np.array(ratio[0]), np.array(ratio))
            bin_edges = hist_ref[1]
            ax_ratio.step(bin_edges, ratio, where="pre", **arr_dict.get("hist_kwargs", {}))

    ax_upper.legend(
        # handles=legend_handles,
        loc=leg_loc,
        frameon=False,
        title=leg_title,
        ncol=leg_ncols,
    )
    # re-do legend, with the first handle kep and the others replaced by the new list
    old_handles, old_labels = ax_upper.get_legend_handles_labels()
    new_handles = old_handles[:1] + legend_handles if ref_dict is not None else legend_handles
    ax_upper.legend(
        handles=new_handles,
        loc=leg_loc,
        frameon=False,
        title=leg_title,
        ncol=leg_ncols,
    )
    ax_upper.set_ylabel("Normalized")

    ax_upper.set_xlim(bins[0], bins[-1])

    if ax_ratio is not None:
        ax_ratio.set_xlim(bins[0], bins[-1])
        ax_upper.set_xticks([])

    if ratio_range is not None:
        ax_ratio.set_ylim(*ratio_range)
    if xlabel is not None:
        if ax_ratio is not None:
            ax_ratio.set_xlabel(xlabel)
        else:
            ax_upper.set_xlabel(xlabel)
    if logy:
        ax_upper.set_yscale("log")
    return hist_curve_dict if return_hist_curve else None


def plot_two_shower_versions(const1, const2, label1="version1", label2="version2", title=None):
    """Plot the constituent and shower features for two shower collections.

    Parameters:
    ----------
    const1 : awkward array
        Constituents of the first shower collection.
    const2 : awkward array
        Constituents of the second shower collection.
    title : str, optional
        Title of the plot.
    """

    showers1 = ak.sum(const1, axis=1)
    showers2 = ak.sum(const2, axis=1)

    fig, axarr = plt.subplots(4, 4, figsize=(12, 8))
    histkwargs = dict(bins=100, density=True, histtype="step")

    part_feats = ["pt", "eta", "phi", "mass"]
    for i, feat in enumerate(part_feats):
        axarr[0, i].hist(ak.flatten(const1[feat]), **histkwargs, label=label1)
        axarr[0, i].hist(ak.flatten(const2[feat]), **histkwargs, label=label1)
        axarr[0, i].set_xlabel(f"Constituent {feat}")
        # plot the difference
        axarr[1, i].hist(
            ak.flatten(const2[feat]) - ak.flatten(const1[feat]),
            **histkwargs,
            label=f"{label2} - {label1}",
        )
        axarr[1, i].set_xlabel(f"Constituent {feat} resolution")

    shower_feats = ["pt", "eta", "phi", "mass"]
    for i, feat in enumerate(shower_feats):
        axarr[2, i].hist(getattr(showers1, feat), **histkwargs, label=label1)
        axarr[2, i].hist(getattr(showers2, feat), **histkwargs, label=label2)
        axarr[2, i].set_xlabel(f"shower {feat}")
        axarr[3, i].hist(
            getattr(showers2, feat) - getattr(showers1, feat),
            **histkwargs,
            label=f"{label2} - {label1}",
        )
        axarr[3, i].set_xlabel(f"shower {feat} resolution")

    axarr[0, 0].legend(frameon=False)
    axarr[1, 0].legend(frameon=False)
    axarr[2, 0].legend(frameon=False)
    axarr[3, 0].legend(frameon=False)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    # plt.show()
    return fig, axarr


def plot_features(
    ak_array_dict,
    names=None,
    label_prefix=None,
    flatten=True,
    histkwargs=None,
    legend_only_on=None,
    legend_kwargs={},
    ax_rows=1,
    decorate_ax_kwargs={},
    bins_dict=None,
    colors=None,
):
    """Plot the features of the constituents or showers.

    Parameters:
    ----------
    ak_array_dict : dict of awkward array
        Dict with {"name": ak.Array, ...} of the constituents or showers to plot.
    names : list of str or dict, optional
        Names of the features to plot. Either a list of names, or a dict of {"name": "label", ...}.
    label_prefix : str, optional
        Prefix for the plot x-axis labels.
    flatten : bool, optional
        Whether to flatten the arrays before plotting. Default is True.
    histkwargs : dict, optional
        Keyword arguments passed to plt.hist.
    legend_only_on : int, optional
        Plot the legend only on the i-th subplot. Default is None.
    legend_kwargs : dict, optional
        Keyword arguments passed to ax.legend.
    ax_rows : int, optional
        Number of rows of the subplot grid. Default is 1.
    decorate_ax_kwargs : dict, optional
        Keyword arguments passed to `decorate_ax`.
    bins_dict : dict, optional
        Dict of {name: bins} for the histograms. `name` has to be the same as the keys in `names`.
    colors : list, optional
        List of colors for the histograms. Has to have the same length as the number of arrays.
        If shorter, the colors will be repeated.
    """

    default_hist_kwargs = {"density": True, "histtype": "step", "bins": 100}

    # setup colors
    if colors is not None:
        if len(colors) < len(ak_array_dict):
            print(
                "Warning: colors list is shorter than the number of arrays. "
                "Will use default colors for remaining ones."
            )
            colors = colors + [f"C{i}" for i in range(len(ak_array_dict) - len(colors))]

    if histkwargs is None:
        histkwargs = default_hist_kwargs
    else:
        histkwargs = default_hist_kwargs | histkwargs

    # create the bins dict
    if bins_dict is None:
        bins_dict = {}
    # loop over all names - if the name is not in the bins_dict, use the default bins
    for name in names:
        if name not in bins_dict:
            bins_dict[name] = histkwargs["bins"]

    # remove default bins from histkwargs
    histkwargs.pop("bins")

    if isinstance(names, list):
        names = {name: name for name in names}

    ax_cols = len(names) // ax_rows + 1

    fig, axarr = plt.subplots(ax_rows, ax_cols, figsize=(3 * ax_cols, 2 * ax_rows))
    axarr = axarr.flatten()

    legend_handles = []
    legend_labels = []

    for i_label, (label, ak_array) in enumerate(ak_array_dict.items()):
        color = colors[i_label] if colors is not None else f"C{i_label}"
        legend_labels.append(label)
        for i, (feat, feat_label) in enumerate(names.items()):
            if flatten:
                values = ak.flatten(getattr(ak_array, feat))
            else:
                values = getattr(ak_array, feat)

            if not isinstance(bins_dict[feat], int):
                values = binclip(values, bins_dict[feat])

            _, _, patches = axarr[i].hist(values, **histkwargs, bins=bins_dict[feat], color=color)
            axarr[i].set_xlabel(
                feat_label if label_prefix is None else f"{label_prefix} {feat_label}"
            )
            if i == 0:
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color=patches[0].get_edgecolor(),
                        lw=patches[0].get_linewidth(),
                        label=label,
                        linestyle=patches[0].get_linestyle(),
                    )
                )

    legend_kwargs["handles"] = legend_handles
    legend_kwargs["labels"] = legend_labels
    legend_kwargs["frameon"] = False
    for i, _ax in enumerate(axarr):
        if legend_only_on is None:
            _ax.legend(**legend_kwargs)
        else:
            if i == legend_only_on:
                _ax.legend(**legend_kwargs)

        plot_utils.decorate_ax(_ax, **decorate_ax_kwargs)

    fig.tight_layout()
    return fig, axarr


def plot_features_pairplot(
    arr,
    names=None,
    pairplot_kwargs={},
    input_type="ak_constituents",
):
    """Plot the features of the constituents or showers using a pairplot.

    Parameters:
    ----------
    arr : awkward array or numpy array
        Constituents or showers.
    part_names : list or dict, optional
        List of names of the features to plot, or dict of {"name": "label", ...}.
    pairplot_kwargs : dict, optional
        Keyword arguments passed to sns.pairplot.
    input_type : str, optional
        Type of the input array. Can be "ak_constituents", "ak_showers", or "np_flat".
        "ak_constituents" is an awkward array of shower constituents of shape `(n_showers, <var>, n_features)`.
        "ak_showers" is an awkward array of showers of shape `(n_showers, n_features)`.
        "np_flat" is a numpy array of shape `(n_entries, n_features)`


    Returns:
    --------
    pairplot : seaborn.axisgrid.PairGrid
        Pairplot object of the features.
    """

    if isinstance(names, list):
        names = {name: name for name in names}

    sns.set_style("dark")
    # create a dataframe from the awkward array
    if input_type == "ak_constituents":
        df = pd.DataFrame(
            {feat_label: ak.flatten(getattr(arr, feat)) for feat, feat_label in names.items()}
        )
    elif input_type == "ak_showers":
        df = pd.DataFrame({feat_label: getattr(arr, feat) for feat, feat_label in names.items()})
    elif input_type == "np_flat":
        df = pd.DataFrame(
            {feat_label: arr[:, i] for i, (feat, feat_label) in enumerate(names.items())}
        )
    else:
        raise ValueError(f"Invalid input_type: {input_type}")
    pairplot = sns.pairplot(df, kind="hist", **pairplot_kwargs)
    plt.show()

    # reset the style
    plt.rcdefaults()

    return pairplot


def plot_shower_features(
    generated_features: ak = None,
    real_features: ak = None,
    colours: list = ["cornflowerblue", "darkorange"],
    labels: list = ["Real", "Generated"],
):
    """Plot the features of the constituents or showers.

    Parameters:
    ----------
    generated_features : awkward array
        Features of the generated showers.
    real_features : awkward array
        Features of the real showers.
    """

    voxel = ak.to_numpy(ak.num(real_features["x"]))
    voxel_gen = ak.to_numpy(ak.num(generated_features["x"]))

    shower_energy = ak.to_numpy(ak.sum(real_features["energy"], axis=1))
    shower_energy_gen = ak.to_numpy(ak.sum(generated_features["energy"], axis=1))

    max_z = find_max_energy_z(real_features["energy"], real_features["z"])
    max_z_gen = find_max_energy_z(generated_features["energy"], generated_features["z"])

    x_zero = ak.to_numpy(get_COG_ak(real_features["x"], real_features["energy"]))
    y_zero = ak.to_numpy(get_COG_ak(real_features["y"], real_features["energy"]))
    z_zero = ak.to_numpy(get_COG_ak(real_features["z"], real_features["energy"]))

    x_zero_gen = ak.to_numpy(get_COG_ak(generated_features["x"], generated_features["energy"]))
    y_zero_gen = ak.to_numpy(get_COG_ak(generated_features["y"], generated_features["energy"]))
    z_zero_gen = ak.to_numpy(get_COG_ak(generated_features["z"], generated_features["energy"]))

    x = ak.flatten(real_features["x"]).to_numpy()
    y = ak.flatten(real_features["y"]).to_numpy()
    z = ak.flatten(real_features["z"]).to_numpy()
    energy = ak.flatten(real_features["energy"]).to_numpy()

    x_gen = ak.flatten(generated_features["x"]).to_numpy()
    y_gen = ak.flatten(generated_features["y"]).to_numpy()
    z_gen = ak.flatten(generated_features["z"]).to_numpy()
    energy_gen = ak.flatten(generated_features["energy"]).to_numpy()

    x_bin_min = min(x) - 1.5
    x_bin_max = max(x) + 2.5
    y_bin_min = x_bin_min
    y_bin_max = x_bin_max
    z_bin_min = x_bin_min
    z_bin_max = x_bin_max

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = GridSpec(2, 3)
    ############################################################
    # First Histogram - Energy Plots
    ############################################################

    bins = np.logspace(np.log(0.1), np.log(max(energy)), 150, base=np.e)
    ax0 = fig.add_subplot(gs[0])
    ax0.set_title("Visible Energy")
    ax0.hist(
        [energy, energy_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=0.5,
        label=labels,
        color=colours,
    )
    wasserstein_dist = wasserstein_distance(energy, energy_gen)
    kl_divergence = KL(energy, energy_gen, bins)

    ax0.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax0.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax0.set_xlabel("Visible energy (MeV)")
    ax0.set_ylabel("a.u.")
    ax0.legend(loc="upper right")
    ax0.set_xscale("log")
    ax0.set_yscale("log")

    # Energy Sum Histogram
    ax1 = fig.add_subplot(gs[3])
    ax1.set_title("Energy Sum")
    data1 = shower_energy
    data2 = shower_energy_gen
    ax1.hist(
        [data1, data2],
        bins=30,
        histtype="step",
        lw=2,
        alpha=1.0,
        label=labels,
        color=colours,
    )
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, 30)
    ax1.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax1.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax1.set_xlabel("Visible energy sum (MeV)")
    ax1.set_ylabel("a.u.")
    ax1.legend(loc="upper right")

    # z-start-layer

    # Create a new figure
    ax2 = fig.add_subplot(gs[4])
    ax2.set_title("z start layer")
    step = math.ceil(z_bin_max / 11)
    bins = np.arange(z_bin_min, z_bin_max)
    ax2.hist(
        [max_z, max_z_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )
    wasserstein_dist = wasserstein_distance(max_z, max_z_gen)
    kl_divergence = KL(max_z, max_z_gen, bins)
    ax2.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax2.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax2.set_xlabel("z")
    ax2.set_ylabel("a.u.")
    ax2.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )  # Set scientific notation for y-axis

    ax2.set_xticks(np.arange(z_bin_min, z_bin_max, step))
    ax2.legend(loc="upper right")

    # Plot for only y-scale logarithmic
    ax3 = fig.add_subplot(gs[1])
    ax3.set_title("Visible Energy")
    ax3.hist(
        [energy, energy_gen],
        bins=150,
        histtype="step",
        lw=2,
        alpha=0.5,
        label=labels,
        color=colours,
    )

    wasserstein_dist = wasserstein_distance(energy, energy_gen)
    kl_divergence = KL(energy, energy_gen, 150)
    ax3.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax3.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)

    ax3.set_xlabel("Visible energy (MeV)")
    ax3.set_ylabel("a.u.")
    ax3.legend(loc="upper right")
    ax3.set_yscale("log")

    # Plot for only x-scale logarithmic
    ax4 = fig.add_subplot(gs[2])
    bins = np.logspace(np.log(0.1), np.log(max(energy)), 150, base=np.e)
    ax4.set_title("Visible Energy")
    ax4.hist(
        [energy, energy_gen],
        bins,
        histtype="step",
        lw=2,
        alpha=0.5,
        label=labels,
        color=colours,
    )

    ax4.set_xlabel("Visible energy (MeV)")
    ax4.set_ylabel("a.u.")
    ax4.legend(loc="upper right")
    ax4.set_xscale("log")
    wasserstein_dist = wasserstein_distance(energy, energy_gen)
    kl_divergence = KL(energy, energy_gen, bins)
    ax4.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax4.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)

    # Number of Hits Histogram
    ax5 = fig.add_subplot(gs[5])
    ax5.set_title("Number of Hits")
    ax5.hist(
        [voxel, voxel_gen], bins=30, histtype="step", lw=2, alpha=1.0, label=labels, color=colours
    )
    ax5.set_xlabel("n_hits")
    ax5.set_ylabel("a.u.")
    ax5.legend(loc="upper right")
    wasserstein_dist = wasserstein_distance(voxel, voxel_gen)
    kl_divergence = KL(voxel, voxel_gen, 30)
    ax5.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax5.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)

    fig.suptitle("Distributions")

    fig.tight_layout()

    ############################################################
    # Second Histogram      ---     x,y,z Distribution and 0th Moment
    ############################################################

    fig_COG = plt.figure(figsize=(18, 12), facecolor="white")
    gs2 = GridSpec(2, 3)

    ax0 = fig_COG.add_subplot(gs2[0])

    average = sum(x_zero) / len(x_zero)
    if average < 1:
        offset = 0.4
    else:
        offset = average * 0.05

    if average < 0:
        bins = np.arange(-average - offset, -average + offset, 0.005)
    else:
        bins = np.arange(average - offset, average + offset, 0.005)

    ax0.set_title("[X] distribution")
    ax0.hist(
        [x_zero, x_zero_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )
    data1 = x_zero
    data2 = x_zero_gen
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, bins)
    ax0.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax0.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax0.set_xlabel("X")
    ax0.set_ylabel("a.u.")
    ax0.legend(loc="upper right")

    ax1 = fig_COG.add_subplot(gs2[1])
    average = sum(y_zero) / len(y_zero)
    if average < 1:
        offset = 0.4
    else:
        offset = average * 0.05

    if average < 0:
        bins = np.arange(-average - offset, -average + offset, 0.005)
    else:
        bins = np.arange(average - offset, average + offset, 0.005)
    ax1.set_title("[Y] distribution")
    ax1.hist(
        [y_zero, y_zero_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    wasserstein_dist = wasserstein_distance(y_zero, y_zero_gen)
    kl_divergence = KL(y_zero, y_zero_gen, bins)
    ax1.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax1.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax1.set_xlabel("Y")
    ax1.set_ylabel("a.u.")
    ax1.legend(loc="upper right")

    average = sum(z_zero) / len(z_zero)
    if average < 1:
        offset = 1.4
    else:
        offset = average * 0.45

    if average < 0:
        bins = np.arange(-average - offset, -average + offset, 0.05)
    else:
        bins = np.arange(average - offset, average + offset, 0.05)
    ax2 = fig_COG.add_subplot(gs2[2])
    ax2.set_title("[Z] distribution")
    ax2.hist(
        [z_zero, z_zero_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    wasserstein_dist = wasserstein_distance(z_zero, z_zero_gen)
    kl_divergence = KL(z_zero, z_zero_gen, bins)
    ax2.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax2.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax2.set_xlabel("Z")
    ax2.set_ylabel("a.u.")
    ax2.legend(loc="upper right")

    # X Distribution
    ax3 = fig_COG.add_subplot(gs2[3])
    ax3.set_title("[x] distribution")
    ax3.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax3.hist(
        [x, x_gen],
        bins=np.arange(x_bin_min, x_bin_max),
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    data1 = x
    data2 = x_gen
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, np.arange(x_bin_min, x_bin_max))
    ax3.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax3.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax3.set_xlabel("[x]")
    ax3.set_ylabel("Number of hits")
    ax3.set_xticks(np.arange(x_bin_min, x_bin_max, step))
    ax3.legend(loc="upper right")

    # Y Distribution
    ax4 = fig_COG.add_subplot(gs2[4])
    ax4.set_title("[y] distribution")
    ax4.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax4.hist(
        [y, y_gen],
        bins=np.arange(y_bin_min, y_bin_max),
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    data1 = y
    data2 = y_gen
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, np.arange(y_bin_min, y_bin_max))
    ax4.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax4.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax4.set_xlabel("[y]")
    ax4.set_ylabel("Number of hits")
    ax4.set_xticks(np.arange(y_bin_min, y_bin_max, step))
    ax4.legend(loc="upper right")

    # Z Distribution
    ax5 = fig_COG.add_subplot(gs2[5])
    ax5.set_title("[z] distribution")
    ax5.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax5.hist(
        [z, z_gen],
        bins=np.arange(z_bin_min, z_bin_max),
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    data1 = z
    data2 = z_gen
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, np.arange(z_bin_min, z_bin_max))
    ax5.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax5.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax5.set_xlabel("[z]")
    ax5.set_ylabel("Number of hits")
    ax5.set_xticks(np.arange(z_bin_min, z_bin_max, step))
    ax5.legend(loc="upper right")
    return fig, fig_COG


def plot_compare_gen_showers(
    feature_sets: list, labels: list = None, colors: list = None, **kwargs
):
    """Plots the features of multiple constituent or shower sets.

    Args:
        feature_sets: A list of dictionaries, each containing awkward arrays for "x", "y", "z", and "energy" features.
        labels: (Optional) A list of labels for the feature sets (defaults to 'Set 1', 'Set 2', etc.).
        colors: (Optional) A list of colors for the feature sets (defaults to a matplotlib colormap).
        kwargs: Additional keyword arguments to pass to the plotting functions.
    """

    num_sets = len(feature_sets)

    if labels is None:
        labels = [f"Set {i + 1}" for i in range(num_sets)]
    if colors is None:
        colors = plt.cm.get_cmap("tab10").colors  # Use matplotlib's colormap

    # Preprocessing & feature extraction
    extracted_features = []
    for features in feature_sets:
        # Filter voxels with energy > 0.1
        mask = features["energy"] > 0.1
        filtered_features = {
            "x": features["x"][mask],
            "y": features["y"][mask],
            "z": features["z"][mask],
            "energy": features["energy"][mask],
        }

        extracted_features.append(
            {
                "voxel": ak.to_numpy(ak.num(filtered_features["x"])),
                "energy": ak.flatten(features["energy"]).to_numpy(),  # Keep all energies here
                "shower_energy": ak.to_numpy(ak.sum(filtered_features["energy"], axis=1)),
                "max_z": find_max_energy_z(filtered_features["energy"], filtered_features["z"]),
                "x_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["x"], filtered_features["energy"])
                ),
                "y_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["y"], filtered_features["energy"])
                ),
                "z_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["z"], filtered_features["energy"])
                ),
                "x": ak.flatten(filtered_features["x"]).to_numpy(),
                "y": ak.flatten(filtered_features["y"]).to_numpy(),
                "z": ak.flatten(filtered_features["z"]).to_numpy(),
                "distance": filtered_features["x"].to_numpy(),  # TODO maybe delete this function
                "energy_filtered": ak.flatten(filtered_features["energy"]).to_numpy(),
            }
        )

    # Plotting (two figures)
    mpl.rcParams["xtick.labelsize"] = 16
    mpl.rcParams["ytick.labelsize"] = 16
    # mpl.rcParams['font.size'] = 28
    mpl.rcParams["font.size"] = 16
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    fig_COG = plt.figure(figsize=(18, 12), facecolor="white")

    # Call the plotting functions, passing the feature sets, labels, and colors
    plot_distributions(fig, extracted_features, labels, colors, **kwargs)
    plot_cog_and_spatial(fig_COG, extracted_features, labels, colors, **kwargs)
    fig_COG.tight_layout()
    fig.tight_layout()

    return fig, fig_COG


def plot_distributions(fig, features_list, labels, colors, **kwargs):
    """Plots the distributions of energy, energy sum, number of hits, and z start layer."""
    gs = fig.add_gridspec(
        5, 3, wspace=0.3, hspace=0.1, height_ratios=[3, 0.8, 0.9, 3, 0.8]
    )  # 3 rows for the different distributions
    # print("Plotting distributions:max(features_list[z])",  max(features_list["z"]))

    # Binning setup (adjust ranges and bins as needed for your data)
    fontsize_labels = 22

    first_features = features_list[0]
    x_max = max(first_features["x"])

    if x_max < 12:  # smaller dataset
        energy_sum = 2000
        energy = 140
        z = 10.5
        n_hits = 400
    else:
        energy_sum = 2000
        energy = 70
        z = 31.5
        n_hits = 1700

    energy_bins = np.logspace(np.log10(0.01), np.log10(energy), 50)  # Logarithmic bins for energy
    energy_sum_bins = np.arange(0, energy_sum, 50)
    max_z_bins = np.arange(-1.5, z, 1)  # Linear bins for z start layer
    voxel_bins = np.arange(0, n_hits, 50)  # The number of hits
    dist_e_bins = np.arange(0, 21, 1)  # The distance

    # Energy Distribution
    ax5 = fig.add_subplot(gs[0, 0])  # vis cell energy x log
    ax0 = fig.add_subplot(gs[0, 1])  # vis cell energy x/y log
    ax4 = fig.add_subplot(gs[0, 2])  # energy over distance
    ax1 = fig.add_subplot(gs[3, 0])  # energy sum
    ax2 = fig.add_subplot(gs[3, 1])  # z start layer
    ax3 = fig.add_subplot(gs[3, 2])  # number of hits

    # looping through all input data to be plottet on the different distributions
    for features, label, color in zip(features_list, labels, colors):
        histtype = "stepfilled" if features is features_list[0] else "step"
        edgecolor = "gray" if histtype == "stepfilled" else color
        linestyle = (
            "--"
            if len(features_list) > 2
            and (
                features is features_list[2]
                or len(features_list) > 3
                and (features is features_list[3])
            )
            else "-"
        )
        alpha = 0.95
        ax0.hist(
            features["energy"],
            bins=energy_bins,
            linestyle=linestyle,
            histtype=histtype,
            edgecolor=edgecolor,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax1.hist(
            features["shower_energy"],
            bins=energy_sum_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax2.hist(
            features["max_z"],
            bins=max_z_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax3.hist(
            features["voxel"],
            bins=voxel_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax4.hist(
            features["distance"],
            bins=dist_e_bins,
            weights=features["energy_filtered"],
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax5.hist(
            features["energy"],
            bins=energy_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
    # ax0.set_xlabel("Energy (MeV)")
    ax0.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")
    ax0.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax0.get_ylim()
    new_ymax = ymax + 62 * ymax
    ax0.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax0_twin = fig.add_subplot(gs[1, 1], sharex=ax0)
    mask = [0.7, 1.3]
    plot_ratios(ax0_twin, features_list, energy_bins, "energy", labels, colors, mask=mask)
    # Add horizontal line at y=1
    ax0_twin.axhline(y=1, color="gray", linestyle="--")
    ax0_twin.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")

    # Set y-axis limits
    ax0_twin.set_ylim(mask)
    ax0_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax0_twin.set_xlabel("visible cell energy [MeV]", fontsize=fontsize_labels)
    ax0_twin.tick_params(axis="y", labelcolor="black")

    # Energy Sum Distribution
    ax1.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax1.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax1.get_ylim()
    new_ymax = ymax + 0.35 * ymax
    ax1.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot
    ax1_twin = fig.add_subplot(gs[4, 0], sharex=ax1)
    plot_ratios(
        ax1_twin, features_list, energy_sum_bins, "shower_energy", labels, colors, mask=mask
    )
    ax1_twin.axhline(y=1, color="gray", linestyle="--")
    # Set y-axis limits
    ax1_twin.set_ylim(mask)
    ax1_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax1_twin.set_xlabel("energy sum [MeV]", fontsize=fontsize_labels)
    ax1_twin.tick_params(axis="y", labelcolor="black")

    # Z Start Layer Distribution
    ax2.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.set_yscale("log")
    ymin, ymax = ax2.get_ylim()
    new_ymax = ymax + 64 * ymax
    ax2.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot
    ax2_twin = fig.add_subplot(gs[4, 1], sharex=ax2)
    mask = [0.6, 1.4]
    plot_ratios(ax2_twin, features_list, max_z_bins, "max_z", labels, colors, mask=mask)
    ax2_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax2_twin.set_ylim(mask)
    ax2_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax2_twin.set_xlabel("shower start layer [layer]", fontsize=fontsize_labels)
    ax2_twin.tick_params(axis="y", labelcolor="black")

    # Number of Hits (Voxel) Distribution
    ax3.set_ylabel("# showers", fontsize=fontsize_labels)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ymin, ymax = ax3.get_ylim()
    new_ymax = ymax + 0.44 * ymax
    ax3.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax3_twin = fig.add_subplot(gs[4, 2], sharex=ax3)
    plot_ratios(ax3_twin, features_list, voxel_bins, "voxel", labels, colors, mask=mask)

    ax3_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax3_twin.set_ylim(mask)
    ax3_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax3_twin.set_xlabel("number of hits", fontsize=fontsize_labels)
    ax3_twin.tick_params(axis="y", labelcolor="black")

    # Energy Distribution per Layer
    # FIXME
    ax4.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax4.set_yscale("log")
    ax4.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax4.get_ylim()
    new_ymax = ymax + 0.18 * ymax
    ax4.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax4_twin = fig.add_subplot(gs[1, 2], sharex=ax4)
    mask = [0.7, 1.3]
    plot_ratios(
        ax4_twin,
        features_list,
        dist_e_bins,
        "distance",
        labels,
        colors,
        mask=mask,
        weights="energy_filtered"
    )

    ax4_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax4_twin.set_ylim(mask)
    ax4_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax4_twin.set_xlabel("radius [pixels]", fontsize=fontsize_labels)
    ax4_twin.tick_params(axis="y", labelcolor="black")

    # Energy Distribution only x-logarithmic
    ax5.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax5.set_xscale("log")
    ax5.tick_params(axis="x", labelbottom=False)
    ax5.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")
    ax5.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax5.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ymin, ymax = ax0.get_ylim()
    new_ymax = ymax + 0.34 * ymax
    ax0.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot
    ax5_twin = fig.add_subplot(gs[1, 0], sharex=ax5)

    plot_ratios(ax5_twin, features_list, energy_bins, "energy", labels, colors, mask=mask)

    ax5_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax5_twin.set_ylim(mask)
    ax5_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax5_twin.set_xlabel("visible cell energy [MeV]", fontsize=fontsize_labels)
    ax5_twin.tick_params(axis="y", labelcolor="black")
    ax5_twin.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")

    # Add legend to the first subplot (energy)
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(colors, labels)
    ]
    # Create the figure
    ax5.legend(handles=legend_elements, loc="upper right")
    ax2.legend(handles=legend_elements, loc="upper right")
    ax3.legend(handles=legend_elements, loc="upper right")
    ax0.legend(handles=legend_elements, loc="upper right")
    ax4.legend(handles=legend_elements, loc="upper right")
    ax1.legend(handles=legend_elements, loc="upper right")

    # Add divergence metrics to the plots
    if len(features_list) == 2:
        for ax in [ax0, ax4, ax5]:
            add_divergence_metrics(
                ax,
                features_list[0]["energy"],
                features_list[1]["energy"],
                energy_bins,
                "energy",
                fontsize=fontsize_labels - 2,
                **kwargs,
            )
        add_divergence_metrics(
            ax1,
            features_list[0]["shower_energy"],
            features_list[1]["shower_energy"],
            energy_sum_bins,
            "energy_sum",
            fontsize=fontsize_labels - 2,
            **kwargs,
        )
        add_divergence_metrics(
            ax2,
            features_list[0]["max_z"],
            features_list[1]["max_z"],
            max_z_bins,
            "max_z",
            fontsize=fontsize_labels - 2,
            **kwargs,
        )
        add_divergence_metrics(
            ax3,
            features_list[0]["voxel"],
            features_list[1]["voxel"],
            voxel_bins,
            "n_hits",
            fontsize=fontsize_labels - 2,
            **kwargs,
        )


def plot_cog_and_spatial(fig_COG, features_list, labels, colors, **kwargs):
    """Plots the COG distributions and spatial distributions of x, y, and z."""
    gs2 = fig_COG.add_gridspec(
        5, 3, wspace=0.3, hspace=0.1, height_ratios=[3, 0.8, 0.9, 3, 0.8]
    )  # 3 rows for the different distributions
    fontsize_labels = 22
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(colors, labels)
    ]

    # COG Distribution Plots
    for i in range(3):
        ax = fig_COG.add_subplot(gs2[0, i])
        ax_twin = fig_COG.add_subplot(gs2[1, i], sharex=ax)
        string = "x_zero" if i == 0 else "y_zero" if i == 1 else "z_zero"
        for features, label, color in zip(features_list, labels, colors):
            histtype = "stepfilled" if features is features_list[0] else "step"
            edgecolor = "gray" if histtype == "stepfilled" else color
            linestyle = (
                "--"
                if len(features_list) > 2
                and (
                    features is features_list[2]
                    or len(features_list) > 3
                    and (features is features_list[3])
                )
                else "-"
            )

            data = features[string]
            average = np.mean(data)

            if average < 7:  # smaller rebinned dataset
                average = 4.5
                # for z 1.4, for x and y 0.4
                offset = 1.4 if i == 2 else 0.4
                # for z 0.2, for x and y 0.05
                steps = 0.2 if i == 2 else 0.05

            else:  # Full resolution dataset
                average = 14.5
                # for z 1.4, for x and y 0.4
                offset = 8 if i == 2 else 0.4
                # for z 0.25, for x and y 0.05
                steps = 0.5 if i == 2 else 0.025

            bins = (
                np.arange(average - offset, average + offset, steps)
                if average >= 0
                else np.arange(-average - offset, -average + offset, steps)
            )
            ax.hist(
                data,
                bins=bins,
                histtype=histtype,
                lw=2,
                alpha=0.8,
                linestyle=linestyle,
                label=label,
                edgecolor=edgecolor,
                color=color,
            )
        mask = [0.5, 1.5]
        plot_ratios(ax_twin, features_list, bins, string, labels, colors, mask=mask)
        ax_twin.set_xlabel(
            f"center of gravity {chr(ord('X')+i)} [voxel]", fontsize=fontsize_labels
        )  # Extract the dimension (X, Y, or Z) from the title
        ax.set_ylabel("# showers", fontsize=fontsize_labels)
        ax.tick_params(axis="x", labelbottom=False)
        ax.legend(handles=legend_elements, loc="upper right")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ymin, ymax = ax.get_ylim()
        new_ymax = ymax + 0.28 * ymax
        ax.set_ylim(ymin, new_ymax)

        ax_twin.axhline(y=1, color="gray", linestyle="--")

        # Set y-axis limits
        ax_twin.set_ylim(mask)
        ax_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
        ax_twin.set_xlabel(
            f"center of gravity {chr(ord('X')+i)} [{'layer' if i == 2 else 'cell'}]",
            fontsize=fontsize_labels,
        )

        ax_twin.tick_params(axis="y", labelcolor="black")
        if len(features_list) == 2:
            add_divergence_metrics(
                ax,
                features_list[0]["x_zero" if i == 0 else "y_zero" if i == 1 else "z_zero"],
                features_list[1]["x_zero" if i == 0 else "y_zero" if i == 1 else "z_zero"],
                bins,
                "X" if i == 0 else "Y" if i == 1 else "Z",
                fontsize=fontsize_labels - 2,
                **kwargs,
            )

    # Spatial Distribution Plots
    for i in range(3):
        ax = fig_COG.add_subplot(gs2[3, i])
        ax_twin = fig_COG.add_subplot(gs2[4, i], sharex=ax)
        string = "x" if i == 0 else "y" if i == 1 else "z"
        for features, label, color in zip(features_list, labels, colors):
            histtype = "stepfilled" if features is features_list[0] else "step"
            edgecolor = "gray" if histtype == "stepfilled" else color
            linestyle = (
                "--"
                if len(features_list) > 2
                and (
                    features is features_list[2]
                    or len(features_list) > 3
                    and (features is features_list[3])
                )
                else "-"
            )
            bins = np.arange(-0.5, 31.5, 1)
            data = features[string]
            ax.hist(
                data,
                bins=bins,
                histtype=histtype,
                lw=2,
                alpha=0.8,
                linestyle=linestyle,
                label=label,
                color=color,
                edgecolor=edgecolor,
            )
        mask = [0.7, 1.3]
        plot_ratios(ax_twin, features_list, bins, string, labels, colors, mask=mask)
        ax_twin.set_xlabel(
            f"spatial distribution {chr(ord('x')+i)} [{'layer' if i == 2 else 'cell'}]",
            fontsize=fontsize_labels,
        )
        ax.set_ylabel("a.u.", fontsize=fontsize_labels)
        ax.legend(handles=legend_elements, loc="upper right")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(axis="x", labelbottom=False)
        ymin, ymax = ax.get_ylim()
        new_ymax = ymax + 0.28 * ymax
        ax.set_ylim(ymin, new_ymax)
        ax_twin.axhline(y=1, color="gray", linestyle="--")

        # Set y-axis limits
        ax_twin.set_ylim(mask)
        ax_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
        ax_twin.tick_params(axis="y", labelcolor="black")
        if len(features_list) == 2:
            add_divergence_metrics(
                ax,
                features_list[0]["x" if i == 0 else "y" if i == 1 else "z"],
                features_list[1]["x" if i == 0 else "y" if i == 1 else "z"],
                bins,
                "x" if i == 0 else "y" if i == 1 else "z",
                fontsize=fontsize_labels - 2,
                **kwargs,
            )


def save_metrics_to_file(file_path, metrics):
    """Save metrics to a JSON file.

    Parameters:
    ----------
    file_path : str
        Path to the JSON file.
    metrics : dict
        Dictionary containing the metrics to save.
    """
    try:
        with open(file_path) as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    data.append(metrics)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def add_divergence_metrics(ax, data1, data2, bins, feature, fontsize, **kwargs):
    """Calculates and adds Wasserstein distance and KL divergence to the plot."""
    wasserstein_dist = wasserstein_distance(data1, data2)
    bins = int(len(bins))
    kl_divergence = quantiled_kl_divergence(data1, data2, bins, False)
    filepath = kwargs.get("filepath", None)
    weights = kwargs.get("weights", None)
    n_data = kwargs.get("n_data", None)
    transfer_learning = kwargs.get("transfer_learning", False)

    if transfer_learning:
        write_distances_to_json(
            kl_divergence, wasserstein_dist, filepath, weights, n_data, feature
        )

    ax.text(
        1.0,
        1.05,
        f"W-distance: {wasserstein_dist:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
    )
    ax.text(
        1.0,
        1.10,
        f"KLD: {kl_divergence:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
    )


def plot_paper_plots(feature_sets: list, labels: list = None, colors: list = None, material: str = None, **kwargs):
    num_sets = len(feature_sets)
    print("Num sets: ", num_sets)
    if labels is None:
        labels = [f"Set {i + 1}" for i in range(num_sets)]
    if colors is None:
        colors = ["lightgrey", "cornflowerblue", "orange","green"][:num_sets]

    if material == "G4_Pb_gamma":
        energy_sum = 2400
    elif material == "G4_Pb_e-":
        energy_sum = 2500
    else:
        energy_sum = 2000

    energy = 70
    if "gamma" in material and "Pb" not in material:
        n_hits = 1700
    elif "W_e-" in material or "Pb_gamma" in material:
        n_hits = 2000
    elif "Pb_e-" in material:
        n_hits = 2700
    else:
        n_hits = 2200


    energy_bins = np.logspace(np.log10(0.01), np.log10(energy), 50)  # Logarithmic bins for energy
    energy_sum_bins = np.arange(0, energy_sum, 75)
    voxel_bins = np.arange(0, n_hits, 50)  # The number of hits
    dist_e_bins = np.arange(0, 21, 1)  # The distance
    if material == "G4_W_gamma":
        bins_cog = np.arange(8, 22, 0.5)
    elif material == "G4_Ta_gamma":
        bins_cog = np.arange(10,25,0.5)
    elif material == "G4_W_e-":
        bins_cog = np.arange(5,17,0.5)
    elif material == "G4_Pb_gamma":
        bins_cog = np.arange(12.5, 28, 0.5)
    elif material == "G4_Pb_e-":
        bins_cog = np.arange(8, 20, 0.5)
    elif material == "G4_Ta_e-":
        bins_cog = np.arange(6., 18, 0.5)
    else:
        bins_cog = np.arange(0, 31.5, 0.5)

    bins_z = np.arange(0, 31.5, 1)


    mpl.rcParams["xtick.labelsize"] = 16
    mpl.rcParams["ytick.labelsize"] = 16
    mpl.rcParams["font.size"] = 16
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = fig.add_gridspec(5, 3, wspace=0.3, hspace=0.1, height_ratios=[3, 0.8, 0.9, 3, 0.8])
    fontsize_labels = 22

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[3, 1])
    ax5 = fig.add_subplot(gs[3, 2])

    ax0_twin = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1_twin = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2_twin = fig.add_subplot(gs[1, 2], sharex=ax2)
    ax3_twin = fig.add_subplot(gs[4, 0], sharex=ax3)
    ax4_twin = fig.add_subplot(gs[4, 1], sharex=ax4)
    ax5_twin = fig.add_subplot(gs[4, 2], sharex=ax5)

    truth_hists, _ , truth_uncertainties, _ = fine_tuning_hists([feature_sets[0]], material=material)
    
    # Plot truth
    alpha = 0.95
    ax0.fill_between(energy_bins, 0, np.append(truth_hists["energy"], 0), 
            step='post', alpha=alpha, color=colors[0], label="Geant4")
    ax1.fill_between(energy_sum_bins, 0, np.append(truth_hists["shower_energy"], 0),
            step='post', alpha=alpha, color=colors[0], label="Geant4")
    ax2.fill_between(voxel_bins, 0, np.append(truth_hists["voxel"], 0),
            step='post', alpha=alpha, color=colors[0] , label="Geant4")
    ax3.fill_between(bins_cog, 0, np.append(truth_hists["z_zero"], 0),
            step='post', alpha=alpha, color=colors[0], label="Geant4")
    ax4.fill_between(bins_z, 0, np.append(truth_hists["hits"], 0),
            step='post', alpha=alpha, color=colors[0], label="Geant4")
    ax5.fill_between(dist_e_bins, 0, np.append(truth_hists["pixel"], 0),
            step='post', alpha=alpha, color=colors[0], label="Geant4")
    
    for i in range(1, num_sets):
        label = labels[i]
        color = colors[i]
        linestyle = "-"
        
    
        _, gen_hists, _ , gen_uncertainties = fine_tuning_hists([feature_sets[0], feature_sets[i]], material=material)
        
        # Plot generated data
        if material == "G4_W_e-" or material == "G4_Ta_e-" or material == "G4_Pb_e-":
            # Step to zero for electrons
            energy_sum_bins_with_zero = np.insert(energy_sum_bins, 0, 0)  
            ax1.step(energy_sum_bins_with_zero, np.concatenate(([0], gen_hists["shower_energy"], [0])),
                    where='post', linestyle=linestyle, lw=2, label=label, color=color)
            voxel_bins_with_zero = np.insert(voxel_bins, 0, 0)  
            ax2.step(voxel_bins_with_zero, np.concatenate(([0], gen_hists["voxel"], [0])),
                    where='post', linestyle=linestyle, lw=2, label=label, color=color) 
        else:
            ax1.step(energy_sum_bins, np.append(gen_hists["shower_energy"], 0),
                    where='post', linestyle=linestyle, lw=2, label=label, color=color)
            ax2.step(voxel_bins, np.append(gen_hists["voxel"], 0),
                    where='post', linestyle=linestyle, lw=2, label=label, color=color)

        ax0.step(energy_bins, np.append(gen_hists["energy"], 0), 
                where='post', linestyle=linestyle, lw=2, label=label, color=color)
        ax3.step(bins_cog, np.append(gen_hists["z_zero"], 0),
                where='post', linestyle=linestyle, lw=2, label=label, color=color)
        ax4.step(bins_z, np.append(gen_hists["hits"], 0),
                where='post', linestyle=linestyle, lw=2, label=label, color=color)
        ax5.step(dist_e_bins, np.append(gen_hists["pixel"], 0),
                where='post', linestyle=linestyle, lw=2, label=label, color=color)

        mask = [0.465, 1.535]
        plot_ratios_np(
            ax0_twin,
            gen=(gen_hists["energy"], gen_uncertainties["energy"]),
            truth=(truth_hists["energy"], truth_uncertainties["energy"]),
            bins=energy_bins,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax1_twin,
            gen=(gen_hists["shower_energy"], gen_uncertainties["shower_energy"]),
            truth=(truth_hists["shower_energy"], truth_uncertainties["shower_energy"]),
            bins=energy_sum_bins,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax2_twin,
            gen=(gen_hists["voxel"], gen_uncertainties["voxel"]),
            truth=(truth_hists["voxel"], truth_uncertainties["voxel"]),
            bins=voxel_bins,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax3_twin,
            gen=(gen_hists["z_zero"], gen_uncertainties["z_zero"]),
            truth=(truth_hists["z_zero"], truth_uncertainties["z_zero"]),
            bins=bins_cog,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax4_twin,
            gen=(gen_hists["hits"], gen_uncertainties["hits"]),
            truth=(truth_hists["hits"], truth_uncertainties["hits"]),
            bins=bins_z,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax5_twin,
            gen=(gen_hists["pixel"], gen_uncertainties["pixel"]),
            truth=(truth_hists["pixel"], truth_uncertainties["pixel"]),
            bins=dist_e_bins,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
    ax0.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax0_twin.set_xlabel("visible cell energy [MeV]", fontsize=fontsize_labels)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(left=0.01)
    ax0.axvspan(0.01, 0.1, ymin=0, ymax=0.73, facecolor="lightgray", alpha=0.2, hatch="/")
    ax0.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax0.get_ylim()
    ax0.set_ylim(ymin, ymax + 1620 * ymax)
    ax0.set_ylim(bottom=0.1)
    ax0.legend(loc="upper right", fontsize=fontsize_labels - 5,ncol=2)#,columnspacing=-2.1)

    ax1.set_ylabel("normalized", fontsize=fontsize_labels)
    ax1_twin.set_xlabel("energy sum [MeV]", fontsize=fontsize_labels)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.set_ylim(bottom=0)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax + 0.45 * ymax)
    ax1.legend(loc="upper right", fontsize=fontsize_labels - 5,ncol=2)#,columnspacing=-2.1)

    ax2.set_ylabel("normalized", fontsize=fontsize_labels)
    ax2_twin.set_xlabel("number of hits", fontsize=fontsize_labels)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax2.set_ylim(bottom=0)
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin, ymax + 0.44 * ymax)
    ax2.legend(loc="upper right", fontsize=fontsize_labels - 5,ncol=2)#,columnspacing=-2.1)

    ax3.set_ylabel("normalized", fontsize=fontsize_labels)
    ax3_twin.set_xlabel("center of gravity Z [layer]", fontsize=fontsize_labels)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax3.set_ylim(bottom=0)
    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim(ymin, ymax + 0.48 * ymax)
    ax3.legend(loc="upper right", fontsize=fontsize_labels - 5,ncol=2)#,columnspacing=-2.1)

    ax4.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax4_twin.set_xlabel("layer", fontsize=fontsize_labels)
    ax4.tick_params(axis="x", labelbottom=False)
    ax4.set_yscale("log")
    ax4.set_xlim(0, 30)
    ax4.set_ylim(bottom=0.1)
    ymin, ymax = ax4.get_ylim()
    ax4.set_ylim(ymin, ymax + 40 * ymax)
    ax4.legend(loc="upper right", fontsize=fontsize_labels - 5,ncol=2)#,columnspacing=-2.1)

    ax5.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax5_twin.set_xlabel("radius [pixels]", fontsize=fontsize_labels)
    ax5.set_yscale("log")
    ax5.set_xlim(0, 21)
    ax5.tick_params(axis="x", labelbottom=False)
    ax5.set_ylim(bottom=0.1)
    ymin, ymax = ax5.get_ylim()
    ax5.set_ylim(ymin, ymax + 40 * ymax)
    ax5.legend(loc="upper right", fontsize=fontsize_labels - 5,ncol=2)#,columnspacing=-2.1)

    for ax_twin, ax, xlim in [(ax0_twin, ax0, None), (ax1_twin, ax1, None), (ax2_twin, ax2, None),
                               (ax3_twin, ax3, None), (ax4_twin, ax4, (0, 30)), (ax5_twin, ax5, (0, 21))]:
        ax_twin.axhline(y=1, color="gray", linestyle="--", lw=1)
        ax_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
        ax_twin.set_ylim([0.5, 1.5])  
        ax_twin.tick_params(axis="y", labelcolor="black")
        if xlim:
            ax_twin.set_xlim(xlim)

    if material == "G4_W_gamma":
        title = r"Tungsten - $\gamma$"
    elif material == "G4_Ta_gamma":
        title = r"Tantalum - $\gamma$"
    elif material == "G4_W_e-":
        title = r"Tungsten - $e^-$"
    elif material == "G4_Pb_gamma":
        title = r"Lead - $\gamma$"
    elif material == "G4_Pb_e-":
        title = r"Lead - $e^-$"
    elif material == "G4_Ta_e-":
        title = r"Tantalum - $e^-$"
    else:
        title = material

    fig.suptitle(f"{title}", fontsize=35, y=0.95)
    return fig

def decode_hits(tokens, energies, grid_size=30, SOS_token=0, EOS_token=27001,PAD_token=27002,ground_truth=False):
    tokens = np.asarray(tokens)
    energies = np.asarray(energies)
    
    # Remove SOS, EOS or PAD
    pixel_mask_tokens = np.array([PAD_token, SOS_token, EOS_token])
    pixel_mask = ~np.isin(tokens, pixel_mask_tokens)
    mask = pixel_mask # must be valid in both pixel and energy - this is taken care of, see generate() function of GPT.py
    
    tokens = tokens[mask] - 1 # tokens are offset by 1
    energies = energies[mask]
    
    # Convert flat token -> (z, y, x)
    z = tokens // (grid_size * grid_size)
    rem = tokens % (grid_size * grid_size)
    y = rem // grid_size
    x = rem % grid_size

    assert energies.any() != -1

    # Safety check
    if np.any(tokens < 0) or np.any(tokens >= grid_size**3):
        raise ValueError("Decoded token out of [0, grid_size^3). Check token definitions.")

    return z,x,y,energies

        

def read_generated(file_path,tokenizer,material_list=["G4_W_gamma","G4_Ta_gamma","G4_Pb_gamma"],
                    num_showers=-1,material="G4_W",apply_correction=False,topk=3,shift=10):
    with h5py.File(file_path,"r") as h5file:
        showers = h5file['showers'][()]

        data_dict = {
            "x": [],
            "y": [],
            "z": [],
            "energy": [],
        }

        data_dict_truth = {
            "x": [],
            "y": [],
            "z": [],
            "energy": []
        }

        if num_showers == -1:
            num_showers = len(showers)

        total_showers = len(showers)

        print(f"Total {material} showers in file: {total_showers}")
        
        for i,shower in enumerate(showers):
            #if i == 10000: break # for testing
            init_E,spatial,energy,spatial_truth,energy_truth,material_index = shower
            mat = material_list[material_index]
            # print(f"Shower {i}: Material index {material_index} -> {mat}")
            if mat != material:
                continue
            
            # Primary decode step
            # Decode indices to x,y,z and filter tokens
            x,y,z,E = decode_hits(spatial,energy) 
            E = tokenizer.de_tokenize(E)
            if apply_correction:
                if E.shape[0] > 5: 
                    top_k = np.argpartition(E, -topk)[-topk:]  # Get indices of top k energies
                    x[top_k] += shift # Shift top k hits back 
                    x = np.clip(x, 0, 29) # Ensure we don't go out of bounds

            xt,yt,zt,Et = decode_hits(spatial_truth,energy_truth)
            
            if i % 5000 == 0 or i == num_showers:
                print(f"Shower #: {i}/{num_showers}, Material: {mat}")

            data_dict["z"].append(x)
            data_dict["x"].append(y)
            data_dict["y"].append(z)
            data_dict["energy"].append(E)
            
            data_dict_truth["z"].append(xt)
            data_dict_truth["x"].append(yt)
            data_dict_truth["y"].append(zt)
            data_dict_truth["energy"].append(Et)

            if i == num_showers:
                break

        ak_array = ak.Array(data_dict)
        ak_array_truth = ak.Array(data_dict_truth)
        return ak_array,ak_array_truth

def make_plots(file_path,tokenizer,materials_to_plot=None,num_showers=-1,material_list=["G4_W","G4_Ta","G4_Pb"],comparison_path=None):
    
    if materials_to_plot is None:
        raise ValueError("materials_to_plot must be provided as a list of material names.")

    os.makedirs("Plots",exist_ok=True)
    filename = file_path.split("/")[-1][:-3]

    for material in materials_to_plot:
        print("Making plots for material:",material)
        if material == "G4_Ta_e-":
            apply_correction = True
            topk = 1
        elif material == "G4_Pb_e-":
            apply_correction = True
            topk = 3
        else:           
            apply_correction = False
            topk = None

        generated_features, ground_truth_features = read_generated(file_path, tokenizer, material_list, num_showers, material, apply_correction=apply_correction, topk=topk)

        if comparison_path and material == "G4_W_gamma":
            print("Loading Omnijet alpha_c features for comparison...")
            omnijet_alpha_c = plot_utils.read_and_filter_energies(comparison_path)
            labels = ["Geant4",r"Omnijet-$\alpha_{c}$", "Ours"]
            colors  = ["lightgrey", "green", "cornflowerblue"]
            # Sample out even sizes:
            min_size = min(len(omnijet_alpha_c), len(ground_truth_features))
            omnijet_alpha_c = omnijet_alpha_c[:min_size]
            ground_truth_features = ground_truth_features[:min_size]
            generated_features = generated_features[:min_size]
            input_features = [ground_truth_features, omnijet_alpha_c, generated_features]
        else:
            labels = ["Geant4", "Ours"]
            colors = ["lightgrey", "cornflowerblue"] 
            input_features = [ground_truth_features, generated_features]

        fig = plot_paper_plots(
           input_features,
           labels=labels,
           colors=colors, material=material
        )


        # fig.tight_layout()
        fig.savefig(f"Plots/{filename}_{material}.pdf", dpi=300)


def map_vocab_to_tokens(tokens, grid_size=30):
    z = tokens // (grid_size * grid_size)
    rem = tokens % (grid_size * grid_size)
    y = rem // grid_size
    x = rem % grid_size

    return x,y,z

def visualize_vocab_LoRA(A,B,W_orig,label):
    os.makedirs("Plots",exist_ok=True)
    os.makedirs("Plots/Vocab_Visualizations",exist_ok=True)

    W  = B @ A + W_orig  # [V, D]
    W_token = W.mean(dim=1)  # [V], signed
    denom = W_token.abs().max() + 1e-12
    W_norm = (W_token / denom).detach().cpu().numpy()  # [V], signed in [-1,1]

    if label == "Pixel":
        W_norm = W_norm[:-1] # exclude PAD tokens        
        plt.tight_layout()
        fig,ax = plt.subplots(1,1,figsize=(6,4))
        ax.bar(np.arange(len(W_norm)),W_norm,color="cornflowerblue",alpha=1.0)
        ax.set_xlabel("Pixel Token Bias")
        ax.set_ylabel("a.u.")
        ax.set_title(f"Vocabulary Token Bias Visualization - {label}")

        grid_size = 30
        tokens_per_layer = grid_size * grid_size  # 900 tokens per layer
        for layer in range(1, 30):  # 30 layers total
            ax.axvline(x=layer * tokens_per_layer, color='red', linestyle='--', linewidth=1, alpha=0.5)
            #ax.text(layer * tokens_per_layer - tokens_per_layer/2, ax.get_ylim()[1]*0.9, f'Layer {layer-1}', color='red', fontsize=8, ha='center')

        plt.tight_layout()
        plt.savefig(f"Plots/Vocab_Visualizations/Vocab_Bias_{label}.pdf", dpi=300)
        plt.close()

    elif label == "Energy":
        W_norm = W_norm[:-1] # exclude PAD tokens
        fig,ax = plt.subplots(1,1,figsize=(6,4))
        #ax.hist(W_norm,bins=50,color="cornflowerblue",alpha=0.7,density=True)
        ax.bar(np.arange(len(W_norm)),W_norm,color="cornflowerblue",alpha=1.0)
        ax.set_xlabel("Energy Token Bias")
        ax.set_ylabel("a.u.")
        ax.set_title(f"Vocabulary Token Bias Visualization - {label}")

        plt.tight_layout()
        plt.savefig(f"Plots/Vocab_Visualizations/Vocab_Bias_{label}.pdf",dpi=300)
        plt.close()


def fine_tuning_hists(feature_sets: list, material: str = None, **kwargs):
    # Preprocessing & feature extraction
    features_list = []
    for features in feature_sets:
        # Filter voxels with energy > 0.1
        mask = features["energy"] > 0.1
        filtered_features = {
            "x": features["x"][mask],
            "y": features["y"][mask],
            "z": features["z"][mask],
            "energy": features["energy"][mask],
        }

        features_list.append(
            {
                "voxel": ak.to_numpy(ak.num(filtered_features["x"])),
                "energy": ak.flatten(features["energy"]).to_numpy(),  # Keep all energies here
                "shower_energy": ak.to_numpy(ak.sum(filtered_features["energy"], axis=1)),
                # "max_z": find_max_energy_z(filtered_features["energy"], filtered_features["z"]),
                "x_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["x"], filtered_features["energy"])
                ),
                "y_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["y"], filtered_features["energy"])
                ),
                "z_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["z"], filtered_features["energy"])
                ),
                "x": ak.flatten(filtered_features["x"]).to_numpy(),
                "y": ak.flatten(filtered_features["y"]).to_numpy(),
                "z": ak.flatten(filtered_features["z"]).to_numpy(),
                "distance": np.mean(
                    sum_energy_per_radial_distance(
                        filtered_features["x"], filtered_features["y"], filtered_features["energy"]
                    ),
                    axis=0,
                ),
                 "distance_uncertainty":
                     sum_energy_per_radial_distance_unc(
                        filtered_features["x"], filtered_features["y"], filtered_features["energy"]),
                "energy_filtered": ak.flatten(filtered_features["energy"]).to_numpy(),
                "energy_per_layer": np.mean(
                    sum_energy_per_layer(filtered_features["z"], filtered_features["energy"]),
                    axis=0,
                ), 
                "energy_per_layer_uncertainty": 
                    sum_energy_per_layer_unc(
                        filtered_features["z"], filtered_features["energy"]),
                "pixel": np.arange(0, 21) + 0.5,
                "hits": np.arange(0, 29) + 0.5,
            }
        )

    if material == "G4_Pb_gamma":
        energy_sum = 2400
    elif material == "G4_Pb_e-":
        energy_sum = 2500
    else:
        energy_sum = 2000

    energy = 70
    if "gamma" in material and "Pb" not in material:
        n_hits = 1700
    elif "W_e-" in material or "Pb_gamma" in material:
        n_hits = 2000
    elif "Pb_e-" in material:
        n_hits = 2700
    else:
        n_hits = 2200


    energy_bins = np.logspace(np.log10(0.01), np.log10(energy), 50)  # Logarithmic bins for energy
    energy_sum_bins = np.arange(0, energy_sum, 75)
    voxel_bins = np.arange(0, n_hits, 50)  # The number of hits
    dist_e_bins = np.arange(0, 21, 1)  # The distance
    if material == "G4_W_gamma":
        bins_cog = np.arange(8, 22, 0.5)
    elif material == "G4_Ta_gamma":
        bins_cog = np.arange(10,25,0.5)
    elif material == "G4_W_e-":
        bins_cog = np.arange(5,17,0.5)
    elif material == "G4_Pb_gamma":
        bins_cog = np.arange(12.5, 28, 0.5)
    elif material == "G4_Pb_e-":
        bins_cog = np.arange(8, 20, 0.5)
    elif material == "G4_Ta_e-":
        bins_cog = np.arange(6., 18, 0.5)
    else:
        bins_cog = np.arange(0, 31.5, 0.5)

    bins_z = np.arange(0, 31.5, 1)

    truth_histograms = {}
    truth_uncertainties = {}
    gen_histograms = {}
    gen_uncertainties = {}
    
    # Assume we pass [ground_truth_features, generated_features] in that order
    for i, features in enumerate(features_list):
        ax0_, _ = np.histogram(features["energy"], bins=energy_bins)
        ax0_unc = np.sqrt(ax0_)  

        counts1, _ = np.histogram(features["shower_energy"], bins=energy_sum_bins)
        ax1_, _ = np.histogram(features["shower_energy"], bins=energy_sum_bins, density=True)
        bin_widths1 = np.diff(energy_sum_bins)
        total1 = np.sum(counts1)
        ax1_unc = np.sqrt(counts1) / (total1 * bin_widths1)
        
        counts2, _ = np.histogram(features["voxel"], bins=voxel_bins)
        ax2_, _ = np.histogram(features["voxel"], bins=voxel_bins, density=True)
        bin_widths2 = np.diff(voxel_bins)
        total2 = np.sum(counts2)
        ax2_unc = np.sqrt(counts2) / (total2 * bin_widths2)
        
        counts3, _ = np.histogram(features["z_zero"], bins=bins_cog)
        ax3_, _ = np.histogram(features["z_zero"], bins=bins_cog, density=True)
        bin_widths3 = np.diff(bins_cog)
        total3 = np.sum(counts3)
        ax3_unc = np.sqrt(counts3) / (total3 * bin_widths3)
              
        ax4_, _ = np.histogram(features["hits"], bins=bins_z, weights=features["energy_per_layer"])
        ax4_unc = np.asarray(features["energy_per_layer_uncertainty"])  # Use the precomputed uncertainty for energy per layer

        ax5_, _ = np.histogram(features["pixel"], bins=dist_e_bins, weights=features["distance"])
        ax5_unc = np.asarray(features["distance_uncertainty"])  # Use the precomputed uncertainty for distance

        # print("Shapes: ", np.asarray(ax4_).shape, ax4_unc.shape, np.asarray(ax5_).shape, ax5_unc.shape)
        
        if i == 0:
            truth_histograms = {
                "energy": ax0_,
                "shower_energy": ax1_,
                "voxel": ax2_,
                "z_zero": ax3_,
                "hits": ax4_,
                "pixel": ax5_
            }
            truth_uncertainties = {
                "energy": ax0_unc,
                "shower_energy": ax1_unc,
                "voxel": ax2_unc,
                "z_zero": ax3_unc,
                "hits": ax4_unc,
                "pixel": ax5_unc
            }
        else:
            gen_histograms = {
                "energy": ax0_,
                "shower_energy": ax1_,
                "voxel": ax2_,
                "z_zero": ax3_,
                "hits": ax4_,
                "pixel": ax5_
            }
            gen_uncertainties = {
                "energy": ax0_unc,
                "shower_energy": ax1_unc,
                "voxel": ax2_unc,
                "z_zero": ax3_unc,
                "hits": ax4_unc,
                "pixel": ax5_unc
            }

    return truth_histograms, gen_histograms, truth_uncertainties, gen_uncertainties

def make_fine_tune_hists(base_dir,tokenizer,material_list=["G4_W","G4_Ta","G4_Pb"],material_to_plot=None,dataset_size="1k",full_path=None):
    
    if material_to_plot is None:
        raise ValueError("materials_to_plot must be provided as a list of material names.")

    if full_path is None:
        base_paths = f"{base_dir}/{material_to_plot}_{dataset_size}"
        file_list = os.listdir(base_paths)
        print("\n")
        truth_hists = []
        gen_hists = []
        gen_uncertainties = []
        truth_uncertainties = []
        for i,file in enumerate(file_list):
            if i == 5: break  # limit to 5 files for testing
            print("============================================")
            file_path = os.path.join(base_paths, file)

            print("Making histograms for material:",material_to_plot, "data size:", dataset_size)
            # read_generated(file_path, tokenizer, material_list, num_showers, material)
            print("Processing file:", file_path)
            generated_features, ground_truth_features = read_generated(file_path, tokenizer, material_list=material_list, material=material_to_plot)
            truth, gen, truth_unc, gen_unc = fine_tuning_hists(
                [ground_truth_features, generated_features],
                material=material_to_plot
            )
            truth_hists.append(truth)
            gen_hists.append(gen)
            gen_uncertainties.append(gen_unc)
            truth_uncertainties.append(truth_unc)
            print("============================================")
            print("\n")

        
        true_ax0 = truth_hists[0]["energy"]
        true_ax1 = truth_hists[0]["shower_energy"]
        true_ax2 = truth_hists[0]["voxel"]
        true_ax3 = truth_hists[0]["z_zero"]
        true_ax4 = truth_hists[0]["hits"]
        true_ax5 = truth_hists[0]["pixel"]

        true_ax0_std = truth_uncertainties[0]["energy"]
        true_ax1_std = truth_uncertainties[0]["shower_energy"]
        true_ax2_std = truth_uncertainties[0]["voxel"]
        true_ax3_std = truth_uncertainties[0]["z_zero"]
        true_ax4_std = truth_uncertainties[0]["hits"]
        true_ax5_std = truth_uncertainties[0]["pixel"]

        gen_ax0 = np.mean([gen_histograms["energy"] for gen_histograms in gen_hists], axis=0)
        #gen_ax0_std = np.std([gen_histograms["energy"] for gen_histograms in gen_hists], axis=0)
        #gen_ax0_std = np.sqrt(np.std([gen_histograms["energy"] for gen_histograms in gen_hists], axis=0)**2 + np.mean([gen_uncertainties["energy"] for gen_uncertainties in gen_hists], axis=0)**2) # Add in quadrature
        
        gen_ax1 = np.mean([gen_histograms["shower_energy"] for gen_histograms in gen_hists], axis=0)
        #gen_ax1_std = np.std([gen_histograms["shower_energy"] for gen_histograms in gen_hists], axis=0)
        #gen_ax1_std = np.sqrt(np.std([gen_histograms["shower_energy"] for gen_histograms in gen_hists], axis=0)**2 + np.mean([gen_uncertainties["shower_energy"] for gen_uncertainties in gen_hists], axis=0)**2)
        
        gen_ax2 = np.mean([gen_histograms["voxel"] for gen_histograms in gen_hists], axis=0)
        #gen_ax2_std = np.std([gen_histograms["voxel"] for gen_histograms in gen_hists], axis=0)
        #gen_ax2_std = np.sqrt(np.std([gen_histograms["voxel"] for gen_histograms in gen_hists], axis=0)**2 + np.mean([gen_uncertainties["voxel"] for gen_uncertainties in gen_hists], axis=0)**2)
        
        gen_ax3 = np.mean([gen_histograms["z_zero"] for gen_histograms in gen_hists], axis=0)
        #gen_ax3_std = np.std([gen_histograms["z_zero"] for gen_histograms in gen_hists], axis=0)
        #gen_ax3_std = np.sqrt(np.std([gen_histograms["z_zero"] for gen_histograms in gen_hists], axis=0)**2 + np.mean([gen_uncertainties["z_zero"] for gen_uncertainties in gen_hists], axis=0)**2)
        
        gen_ax4 = np.mean([gen_histograms["hits"] for gen_histograms in gen_hists], axis=0)
        #gen_ax4_std = np.std([gen_histograms["hits"] for gen_histograms in gen_hists], axis=0)
        #gen_ax4_std = np.sqrt(np.std([gen_histograms["hits"] for gen_histograms in gen_hists], axis=0)**2 + np.mean([gen_uncertainties["hits"] for gen_uncertainties in gen_hists], axis=0)**2)
        
        gen_ax5 = np.mean([gen_histograms["pixel"] for gen_histograms in gen_hists], axis=0)
        #gen_ax5_std = np.std([gen_histograms["pixel"] for gen_histograms in gen_hists], axis=0)
        #gen_ax5_std = np.sqrt(np.std([gen_histograms["pixel"] for gen_histograms in gen_hists], axis=0)**2 + np.mean([gen_uncertainties["pixel"] for gen_uncertainties in gen_hists], axis=0)**2)

        t1 = np.mean([gen_unc["energy"]**2 for gen_unc in gen_uncertainties], axis=0)
        t2 = np.std([gen_histograms["energy"] for gen_histograms in gen_hists], axis=0)**2
        min_ = min(len(t1), len(t2))
        gen_ax0_std = np.sqrt(t1[:min_] + t2[:min_])

        t1 = np.mean([gen_unc["shower_energy"]**2 for gen_unc in gen_uncertainties], axis=0)
        t2 = np.std([gen_histograms["shower_energy"] for gen_histograms in gen_hists], axis=0)**2
        min_ = min(len(t1), len(t2))
        gen_ax1_std = np.sqrt(t1[:min_] + t2[:min_])
        t1 = np.mean([gen_unc["voxel"]**2 for gen_unc in gen_uncertainties], axis=0)
        t2 = np.std([gen_histograms["voxel"] for gen_histograms in gen_hists], axis=0)**2
        min_ = min(len(t1), len(t2))
        gen_ax2_std = np.sqrt(t1[:min_] + t2[:min_])

        t1 = np.mean([gen_unc["z_zero"]**2 for gen_unc in gen_uncertainties], axis=0)
        t2 = np.std([gen_histograms["z_zero"] for gen_histograms in gen_hists], axis=0)**2
        min_ = min(len(t1), len(t2))
        gen_ax3_std = np.sqrt(t1[:min_] + t2[:min_])

        t1 = np.mean([gen_unc["hits"]**2 for gen_unc in gen_uncertainties], axis=0)
        t2 = np.std([gen_histograms["hits"] for gen_histograms in gen_hists], axis=0)**2
        min_ = min(len(t1), len(t2))
        gen_ax4_std = np.sqrt(t1[:min_] + t2[:min_])

        t1 = np.mean([gen_unc["pixel"]**2 for gen_unc in gen_uncertainties], axis=0)
        t2 = np.std([gen_histograms["pixel"] for gen_histograms in gen_hists], axis=0)**2
        min_ = min(len(t1), len(t2))
        gen_ax5_std = np.sqrt(t1[:min_] + t2[:min_])

        return {
            "truth": {
                "energy": (true_ax0, true_ax0_std),
                "shower_energy": (true_ax1, true_ax1_std),
                "voxel": (true_ax2, true_ax2_std),
                "z_zero": (true_ax3, true_ax3_std),
                "hits": (true_ax4, true_ax4_std),
                "pixel": (true_ax5, true_ax5_std)
            },
            "gen": {
                "energy": (gen_ax0, gen_ax0_std),
                "shower_energy": (gen_ax1, gen_ax1_std),
                "voxel": (gen_ax2, gen_ax2_std),
                "z_zero": (gen_ax3, gen_ax3_std),
                "hits": (gen_ax4, gen_ax4_std),
                "pixel": (gen_ax5, gen_ax5_std)
            }
        }

    else:
        print("============================================")
        print("Making histograms for material:",material_to_plot, "data size:", dataset_size)
        print("Processing full path file:", full_path)
        generated_features, ground_truth_features = read_generated(full_path, tokenizer, material_list=material_list, material=material_to_plot)
        truth,gen, truth_uncertainties, gen_uncertainties = fine_tuning_hists(
            [ground_truth_features, generated_features],
            material=material_to_plot
        )

        print("============================================")
        print("\n")

        true_ax0 = truth["energy"]
        true_ax1 = truth["shower_energy"]
        true_ax2 = truth["voxel"]
        true_ax3 = truth["z_zero"]
        true_ax4 = truth["hits"]
        true_ax5 = truth["pixel"]

        true_ax0_std = truth_uncertainties["energy"]
        true_ax1_std = truth_uncertainties["shower_energy"]
        true_ax2_std = truth_uncertainties["voxel"] 
        true_ax3_std = truth_uncertainties["z_zero"]
        true_ax4_std = truth_uncertainties["hits"]
        true_ax5_std = truth_uncertainties["pixel"]

        gen_ax0 = gen["energy"]
        gen_ax1 = gen["shower_energy"]
        gen_ax2 = gen["voxel"]
        gen_ax3 = gen["z_zero"]
        gen_ax4 = gen["hits"]
        gen_ax5 = gen["pixel"]

        gen_ax0_std =  gen_uncertainties["energy"]
        gen_ax1_std =  gen_uncertainties["shower_energy"]
        gen_ax2_std =  gen_uncertainties["voxel"]
        gen_ax3_std =  gen_uncertainties["z_zero"]
        gen_ax4_std =  gen_uncertainties["hits"]
        gen_ax5_std =  gen_uncertainties["pixel"]
       

        return {
            "truth": {
                "energy": (true_ax0, true_ax0_std),
                "shower_energy": (true_ax1, true_ax1_std),
                "voxel": (true_ax2, true_ax2_std),
                "z_zero": (true_ax3, true_ax3_std),
                "hits": (true_ax4, true_ax4_std),
                "pixel": (true_ax5, true_ax5_std)
            },
            "gen": {
                "energy": (gen_ax0, gen_ax0_std),  
                "shower_energy": (gen_ax1, gen_ax1_std),
                "voxel": (gen_ax2, gen_ax2_std),
                "z_zero": (gen_ax3, gen_ax3_std),
                "hits": (gen_ax4, gen_ax4_std),
                "pixel": (gen_ax5, gen_ax5_std)
            }
        }




def plot_fine_tune_comparison(datasets, material_list, material, dataset_sizes, output_dir="FineTuningStudies"):  
    if material == "G4_Pb_gamma":
        energy_sum = 2400
    elif material == "G4_Pb_e-":
        energy_sum = 2500
    else:
        energy_sum = 2000

    energy = 70
    if "gamma" in material and "Pb" not in material:
        n_hits = 1700
    elif "W_e-" in material or "Pb_gamma" in material:
        n_hits = 2000
    elif "Pb_e-" in material:
        n_hits = 2700
    else:
        n_hits = 2200



    energy_bins = np.logspace(np.log10(0.01), np.log10(energy), 50)  # Logarithmic bins for energy
    energy_sum_bins = np.arange(0, energy_sum, 75)
    voxel_bins = np.arange(0, n_hits, 50)  # The number of hits
    dist_e_bins = np.arange(0, 21, 1)  # The distance
    if material == "G4_W_gamma":
        bins_cog = np.arange(8, 22, 0.5)
    elif material == "G4_Ta_gamma":
        bins_cog = np.arange(10,25,0.5)
    elif material == "G4_W_e-":
        bins_cog = np.arange(5,17,0.5)
    elif material == "G4_Pb_gamma":
        bins_cog = np.arange(12.5, 28, 0.5)
    elif material == "G4_Pb_e-":
        bins_cog = np.arange(8, 20, 0.5)
    elif material == "G4_Ta_e-":
        bins_cog = np.arange(6., 18, 0.5)
    else:
        bins_cog = np.arange(0, 31.5, 0.5)

    bins_z = np.arange(0, 31.5, 1)


    energy_bins_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    energy_sum_bins_centers = (energy_sum_bins[:-1] + energy_sum_bins[1:]) / 2
    voxel_bins_centers = (voxel_bins[:-1] + voxel_bins[1:]) / 2
    dist_e_bins_centers = (dist_e_bins[:-1] + dist_e_bins[1:]) / 2
    bins_cog_centers = (bins_cog[:-1] + bins_cog[1:]) / 2   
    bins_z_centers = (bins_z[:-1] + bins_z[1:]) / 2

    labels = [f"{size}" for size in dataset_sizes]
    colors = ["cornflowerblue", "darkorange", "green", "purple"][:len(datasets)]

    mpl.rcParams["xtick.labelsize"] = 16
    mpl.rcParams["ytick.labelsize"] = 16
    mpl.rcParams["font.size"] = 16
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = fig.add_gridspec(5, 3, wspace=0.3, hspace=0.1, height_ratios=[3, 0.8, 0.9, 3, 0.8])
    fontsize_labels = 22

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[3, 1])
    ax5 = fig.add_subplot(gs[3, 2])

    ax0_twin = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1_twin = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2_twin = fig.add_subplot(gs[1, 2], sharex=ax2)
    ax3_twin = fig.add_subplot(gs[4, 0], sharex=ax3)
    ax4_twin = fig.add_subplot(gs[4, 1], sharex=ax4)
    ax5_twin = fig.add_subplot(gs[4, 2], sharex=ax5)

    for i, (key, dataset) in enumerate(datasets.items()):
        truth_hists = dataset["truth"]
        gen_hists = dataset["gen"]
        label = labels[i]
        color = colors[i]
        # linestyle = "-" if i == 0 else "--"  # First is solid (Geant4), rest are dashed
        linestyle = "-"  
        # Plot truth only on first iteration
        alpha = 0.95
        if i == 0:
            if material == "G4_W_e-" or material == "G4_Ta_e-" or material == "G4_Pb_e-":
                # Step to zero for electrons
                    # Plot truth
                
                #ax0.fill_between(energy_bins, 0, np.append(truth_hists["energy"], 0), 
                #        step='post', alpha=alpha, color=colors[0], label="Geant4")
                energy_sum_bins_with_zero = np.insert(energy_sum_bins, 0, 0)  
                ax1.fill_between(energy_sum_bins_with_zero, np.concatenate(([0], truth_hists["shower_energy"][0], [0])),
                        step='post', label="Geant4", color="lightgrey", alpha=alpha)
                voxel_bins_with_zero = np.insert(voxel_bins, 0, 0)  
                ax2.fill_between(voxel_bins_with_zero, np.concatenate(([0], truth_hists["voxel"][0], [0])),
                        step='post', label="Geant4", color="lightgrey", alpha=alpha)
            else:
                ax1.fill_between(energy_sum_bins, np.append(truth_hists["shower_energy"][0], 0),
                        step='post', label="Geant4", color="lightgrey",alpha=alpha)
                
                ax2.fill_between(voxel_bins, np.append(truth_hists["voxel"][0], 0),
                        step='post', label="Geant4", color="lightgrey",alpha=alpha)
                
            ax0.fill_between(energy_bins, np.append(truth_hists["energy"][0], 0), 
                    step='post', label="Geant4", color="lightgrey", alpha=alpha)
            
            ax3.fill_between(bins_cog, np.append(truth_hists["z_zero"][0], 0),
                    step='post', label="Geant4", color="lightgrey", alpha=alpha)
            
            ax4.fill_between(bins_z, np.append(truth_hists["hits"][0], 0),
                    step='post', label="Geant4", color="lightgrey", alpha=alpha)
            
            ax5.fill_between(dist_e_bins, np.append(truth_hists["pixel"][0], 0),
                    step='post', label="Geant4", color="lightgrey", alpha=alpha)
        
        # Plot generated data
        if material == "G4_W_e-" or material == "G4_Ta_e-" or material == "G4_Pb_e-":
            # Step to zero for electrons
            energy_sum_bins_with_zero = np.insert(energy_sum_bins, 0, 0)  
            ax1.step(energy_sum_bins_with_zero, np.concatenate(([0], gen_hists["shower_energy"][0], [0])),
                    where='post', linestyle=linestyle, lw=2, label=label, color=color)
            voxel_bins_with_zero = np.insert(voxel_bins, 0, 0)  
            ax2.step(voxel_bins_with_zero, np.concatenate(([0], gen_hists["voxel"][0], [0])),
                    where='post', linestyle=linestyle, lw=2, label=label, color=color) 
        else:
            ax1.step(energy_sum_bins, np.append(gen_hists["shower_energy"][0], 0),
                    where='post', linestyle=linestyle, lw=2, label=label, color=color)
            ax2.step(voxel_bins, np.append(gen_hists["voxel"][0], 0),
                    where='post', linestyle=linestyle, lw=2, label=label, color=color)

        ax0.step(energy_bins, np.append(gen_hists["energy"][0], 0), 
                where='post', linestyle=linestyle, lw=2, label=label, color=color)
        ax3.step(bins_cog, np.append(gen_hists["z_zero"][0], 0),
                where='post', linestyle=linestyle, lw=2, label=label, color=color)
        ax4.step(bins_z, np.append(gen_hists["hits"][0], 0),
                where='post', linestyle=linestyle, lw=2, label=label, color=color)
        ax5.step(dist_e_bins, np.append(gen_hists["pixel"][0], 0),
                where='post', linestyle=linestyle, lw=2, label=label, color=color)

        # mask = [0.725, 1.275]
        mask = [0.465, 1.535] 

        plot_ratios_np(
            ax0_twin,
            gen=gen_hists["energy"],  # Pass both mean and std for error bars
            truth=truth_hists["energy"],
            bins=energy_bins,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax1_twin,
            gen=gen_hists["shower_energy"],  # Pass both mean and std for error bars
            truth=truth_hists["shower_energy"],
            bins=energy_sum_bins,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax2_twin,
            gen=gen_hists["voxel"],  # Pass both mean and std for error bars
            truth=truth_hists["voxel"],
            bins=voxel_bins,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax3_twin,
            gen=gen_hists["z_zero"],  # Pass both mean and std for error bars
            truth=truth_hists["z_zero"],
            bins=bins_cog,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax4_twin,
            gen=gen_hists["hits"],  # Pass both mean and std for error bars
            truth=truth_hists["hits"],
            bins=bins_z,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )
        plot_ratios_np(
            ax5_twin,
            gen=gen_hists["pixel"],  # Pass both mean and std for error bars
            truth=truth_hists["pixel"],
            bins=dist_e_bins,
            labels=labels,
            color=color,
            mask=mask,
            i=i
        )

    ax0.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax0_twin.set_xlabel("visible cell energy [MeV]", fontsize=fontsize_labels)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(left=0.01)
    ax0.axvspan(0.01, 0.1, ymin=0, ymax=0.73, facecolor="lightgray", alpha=0.2, hatch="/")
    ax0.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax0.get_ylim()
    ax0.set_ylim(ymin, ymax + 1620 * ymax)
    ax0.set_ylim(bottom=0.1)
    ax0.legend(loc="upper right", fontsize=fontsize_labels - 4,ncol=2)#,columnspacing=0.2)

    ax1.set_ylabel("normalized", fontsize=fontsize_labels)
    ax1_twin.set_xlabel("energy sum [MeV]", fontsize=fontsize_labels)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.set_ylim(bottom=0)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax + 0.45 * ymax)
    ax1.legend(loc="upper right", fontsize=fontsize_labels - 4,ncol=2)#,columnspacing=0.2)

    ax2.set_ylabel("normalized", fontsize=fontsize_labels)
    ax2_twin.set_xlabel("number of hits", fontsize=fontsize_labels)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax2.set_ylim(bottom=0)
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin, ymax + 0.44 * ymax)
    ax2.legend(loc="upper right", fontsize=fontsize_labels - 4,ncol=2)#,columnspacing=0.2)

    ax3.set_ylabel("normalized", fontsize=fontsize_labels)
    ax3_twin.set_xlabel("center of gravity Z [layer]", fontsize=fontsize_labels)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax3.set_ylim(bottom=0)
    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim(ymin, ymax + 0.48 * ymax)
    ax3.legend(loc="upper right", fontsize=fontsize_labels - 4,ncol=2)#,columnspacing=0.2)

    ax4.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax4_twin.set_xlabel("layer", fontsize=fontsize_labels)
    ax4.tick_params(axis="x", labelbottom=False)
    ax4.set_yscale("log")
    ax4.set_xlim(0, 30)
    ax4.set_ylim(bottom=0.1)
    ymin, ymax = ax4.get_ylim()
    ax4.set_ylim(ymin, ymax + 40 * ymax)
    ax4.legend(loc="upper right", fontsize=fontsize_labels - 4,ncol=2)#,columnspacing=0.2)

    ax5.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax5_twin.set_xlabel("radius [pixels]", fontsize=fontsize_labels)
    ax5.set_yscale("log")
    ax5.set_xlim(0, 21)
    ax5.tick_params(axis="x", labelbottom=False)
    ax5.set_ylim(bottom=0.1)
    ymin, ymax = ax5.get_ylim()
    ax5.set_ylim(ymin, ymax + 40 * ymax)
    ax5.legend(loc="upper right", fontsize=fontsize_labels - 4,ncol=2)#,columnspacing=0.2)

    for ax_twin, ax, xlim in [(ax0_twin, ax0, None), (ax1_twin, ax1, None), (ax2_twin, ax2, None),
                               (ax3_twin, ax3, None), (ax4_twin, ax4, (0, 30)), (ax5_twin, ax5, (0, 21))]:
        ax_twin.axhline(y=1, color="gray", linestyle="--", lw=1)
        ax_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
        ax_twin.set_ylim([0.5, 1.5])  
        ax_twin.tick_params(axis="y", labelcolor="black")
        if xlim:
            ax_twin.set_xlim(xlim)

    fig.suptitle(f"Material: {material}", fontsize=22)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/fine_tune_comparison_{material}.pdf", dpi=300)
    plt.close(fig)



def plot_bias_comp(file_path,tokenizer,materials_to_plot=None,num_showers=-1,material_list=["G4_W","G4_Ta","G4_Pb"],comparison_path=None):
    
    if materials_to_plot is None:
        raise ValueError("materials_to_plot must be provided as a list of material names.")

    os.makedirs("Plots",exist_ok=True)
    filename = file_path.split("/")[-1][:-3]

    for material in materials_to_plot:
        print("Making plots for material:",material)
        if material == "G4_Pb_e-":
            topk = 3
            shift = 10
        elif material == "G4_Ta_e-":
            topk = 1
        else:
            raise ValueError(f"Unexpected material: {material}. Expected 'G4_Pb_e-' or 'G4_Ta_e-'.")

        corrected_generated_features, ground_truth_features = read_generated(file_path, tokenizer, material_list, num_showers, material,
                                                                            apply_correction=True,topk=topk)
        generated_features,_ = read_generated(file_path, tokenizer, material_list, num_showers, material,apply_correction=False)

        labels = ["Geant4", "Uncalibrated","Calibrated"]
        colors = ["lightgrey", "green", "cornflowerblue"] 
        input_features = [ground_truth_features, generated_features,corrected_generated_features]

        fig = plot_paper_plots(
           input_features,
           labels=labels,
           colors=colors, material=material
        )


        # fig.tight_layout()
        fig.savefig(f"Plots/{filename}_{material}_bias_comparison.pdf", dpi=300)


# def make_interactive_plots(event_dict):
#     # Event dict -> {"material"} -> {"x": [], "y": [], "z": [], "E": [],"init_E": float}
#     materials = list(event_dict.keys())
#     num_mats = len(materials)
    
#     # Calculate grid dimensions
#     cols = 3
#     rows = (num_mats + cols - 1) // cols  # Ceiling division
    
#     # Create specs as a 2D list matching (rows x cols)
#     specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    
#     names = [f"{mat}: {event_dict[mat]['init_E']:.2f} GeV - Sum: {event_dict[mat]['E'].sum():.2f} MeV" for mat in materials]
#     print(names)

#     fig = make_subplots(
#         rows=rows, cols=cols,
#         subplot_titles=names,
#         specs=specs
#     )

#     # Add a trace for each material
#     for idx, mat in enumerate(materials):
#         # Calculate row and col from flat index
#         row = idx // cols + 1
#         col = idx % cols + 1
#         energy_sum = np.sum(event_dict[mat]['E'])
        
#         data = event_dict[mat]
        
#         # Normalize energies for better color mapping
#         energy_min = np.min(data['E'])
#         energy_max = np.max(data['E'])
#         energy_normalized = (data['E'] - energy_min) / (energy_max - energy_min + 1e-6)
        
#         fig.add_trace(
#             go.Scatter3d(
#                 x=data['x'], y=data['y'], z=data['z'],
#                 mode='markers',
#                 marker=dict(
#                     size=5,
#                     color=energy_normalized,  # Use normalized energy for better contrast
#                     colorscale='Viridis',
#                     showscale=(idx == 0),  # Show colorbar only on first plot
#                     colorbar=dict(
#                         title="Energy (MeV)",
#                         thickness=15,
#                         len=0.7
#                     ),
#                     opacity=0.8,
#                     line=dict(width=0)  # No edge lines for cleaner look
#                 ),
#                 text=[f"E: {e:.2f} MeV" for e in data['E']],  # Hover text
#                 hovertemplate="<b>Position</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>%{text}<extra></extra>",
#                 name=mat
#             ),
#             row=row, col=col
#         )

#     # Set up camera for each subplot with 45-degree tilt
#     # Camera positioned to look at Z axis at 45 degrees
#     camera = dict(
#         eye=dict(x=1.5, y=1.5, z=1.5)  # 45-degree angle viewing
#     )
    
#     # Build dynamic scene updates based on number of materials
#     layout_updates = {
#         'title_text': "Material Event Analysis - 3D Shower Visualization",
#         'height': 500 * rows, 
#         'width': 600 * cols,
#         'showlegend': False
#     }
    
#     # Add scene camera updates for all subplots
#     for i in range(1, num_mats + 1):
#         scene_name = 'scene' if i == 1 else f'scene{i}'
#         layout_updates[scene_name] = dict(
#             camera=camera,
#             xaxis_title="X [cells]",
#             yaxis_title="Y [cells]",
#             zaxis_title="Z [layers]",
#             xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#             yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#             zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
#         )
    
#     fig.update_layout(**layout_updates)
    
#     fig.show()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate a colormap to use only a portion of the color range."""
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def make_interactive_plots(event_dict):
    """
    Create interactive 3D Plotly visualization with voxel-style cubes,
    matching the matplotlib reference style with proper z-axis orientation.
    """
    materials = list(event_dict.keys())
    num_mats = len(materials)
    
    # Calculate grid dimensions
    cols = 3
    rows = (num_mats + cols - 1) // cols
    
    # Create specs as a 2D list matching (rows x cols)
    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    
    names = [f"{mat}: {event_dict[mat]['init_E']:.2f} GeV - Sum: {event_dict[mat]['E'].sum():.2f} MeV" 
             for mat in materials]
    print(names)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=names,
        specs=specs,
        horizontal_spacing=0.002,
        vertical_spacing=0.002
    )

    # Prepare colormap
    cmap = truncate_colormap(mpl.cm.jet, 0.0, 0.7)

    # Add a trace for each material
    for idx, mat in enumerate(materials):
        # Calculate row and col from flat index
        row = idx // cols + 1
        col = idx % cols + 1
        
        data = event_dict[mat]
        
        # Normalize energies for logarithmic opacity scaling
        energy_min = np.min(data['E'])
        energy_max = np.max(data['E'])
        
        xL, yL, zL, cL = [], [], [], []
        colors_list = []
        
        # Collect voxel positions and energies
        for x, y, z, e in zip(data['x'], data['y'], data['z'], data['E']):
            xL.append(x)  
            yL.append(y)  
            zL.append(z)  
            cL.append(e)
            
            # Get color from normalized energy
            norm = mpl.colors.LogNorm(vmin=energy_min, vmax=energy_max)
            norm_val = norm(e)
            color_rgba = cmap(norm_val)
            
            # Calculate opacity based on log scale
            norm_max = energy_max
            alp2 = 0.1 + 0.9 * np.log(e * 10) / np.log(norm_max * 10)
            alp2 = float(np.clip(alp2, 0.1, 0.95))
            
            # ✅ Bake opacity into RGBA color string
            rgba_str = f'rgba({int(color_rgba[0]*255)},{int(color_rgba[1]*255)},{int(color_rgba[2]*255)},{alp2})'
            colors_list.append(rgba_str)
        
        xL = np.array(xL)
        yL = np.array(yL)
        zL = np.array(zL)
        cL = np.array(cL)
        
        fig.add_trace(
            go.Scatter3d(
                x=xL, y=yL, z=zL,
                mode='markers',
                marker=dict(
                    size=6,
                    color=colors_list,  
                    line=dict(width=0)
                ),
                text=[f"E: {e:.2f} MeV" for e in cL],
                hovertemplate="<b>Voxel</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>%{text}<extra></extra>",
                name=mat,
                showlegend=False
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter3d(
                x=[15, 15],  # x stays at 15
                y=[15, 15],  # y stays at 15
                z=[0, 30],   # z goes from 0 to 30
                mode='lines',
                line=dict(
                    color='black',
                    width=4,
                    dash='solid'
                ),
                hoverinfo='skip',
                name='z-axis',
                showlegend=False
            ),
            row=row, col=col
        )

    # Set up camera for each subplot
    camera = dict(
        eye=dict(x=2.0, y=1.5, z=2.0),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=1, z=0)
    )
    
    # Build dynamic scene updates based on number of materials
    layout_updates = {
        'title_text': "Material Event Analysis - 3D Shower Visualization",
        'height': 500 * rows, 
        'width': 700 * cols,
        'showlegend': False
    }
    
    for i in range(1, num_mats + 1):
        scene_name = 'scene' if i == 1 else f'scene{i}'
        layout_updates[scene_name] = dict(
            camera=camera,
            xaxis=dict(
                title="x [cells]",
                title_font=dict(size=18),  # ✅ Use title_font instead of titlefont
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, 30],
                tickfont=dict(size=14)  # ✅ Larger tick font
            ),
            yaxis=dict(
                title="y [cells]",
                title_font=dict(size=18),  # ✅ Use title_font instead of titlefont
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, 30],
                tickfont=dict(size=14)  # ✅ Larger tick font
            ),
            zaxis=dict(
                title="z [layers]",
                title_font=dict(size=18),  # ✅ Use title_font instead of titlefont
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, 30],
                tickfont=dict(size=14)  # ✅ Larger tick font
            ),
            aspectmode='cube'
        )
    
    fig.update_layout(**layout_updates)
    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10)  # Minimal margins
    )
    fig.show()

def make_animated_event_viewer(event_dict, output_dir="Generations", sort_by="z"):
    """
    Create an interactive animated event viewer showing shower development.
    
    Parameters:
    -----------
    event_dict : dict
        Dictionary with material names as keys, each containing 'x', 'y', 'z', 'E' arrays
    output_dir : str
        Directory to save the HTML file
    sort_by : str
        "z" for z-position progression, "energy" for energy-ordered hits
    """
    os.makedirs(output_dir, exist_ok=True)
    
    materials = list(event_dict.keys())
    num_mats = len(materials)
    
    cols = 3
    rows = (num_mats + cols - 1) // cols
    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    
    names = [f"{mat}: {event_dict[mat]['init_E']:.2f} GeV - Sum: {event_dict[mat]['E'].sum():.2f} MeV" 
             for mat in materials]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=names,
        specs=specs,
        horizontal_spacing=0.002,
        vertical_spacing=0.002
    )
    
    cmap = truncate_colormap(mpl.cm.jet, 0.0, 0.7)
    camera = dict(
        eye=dict(x=1.5, y=1.2, z=1.5),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=1, z=0)
    )
    
    # ===== CREATE FRAMES BASED ON SORT_BY =====
    z_frames = []
    energy_frames = []
    
    if sort_by == "z":
        # Z-progression: show hits layer by layer
        z_max = int(max([max(event_dict[mat]['z']) for mat in materials]))
        z_layers = np.arange(0, z_max + 1)
        
        for z_layer in z_layers:
            frame_data = []
            for mat in materials:
                data = event_dict[mat]
                
                # Filter hits up to current z layer
                mask = data['z'] <= z_layer
                x_filtered = data['x'][mask]
                y_filtered = data['y'][mask]
                z_filtered = data['z'][mask]
                E_filtered = data['E'][mask]
                
                if len(E_filtered) == 0:
                    frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', showlegend=False))
                    frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', showlegend=False))
                    continue
                
                energy_min = np.min(data['E'])
                energy_max = np.max(data['E'])
                norm = mpl.colors.LogNorm(vmin=energy_min, vmax=energy_max)
                
                colors_list = []
                for e in E_filtered:
                    norm_val = norm(e)
                    color_rgba = cmap(norm_val)
                    alp2 = 0.1 + 0.9 * np.log(e * 10) / np.log(energy_max * 10)
                    alp2 = float(np.clip(alp2, 0.1, 0.95))
                    rgba_str = f'rgba({int(color_rgba[0]*255)},{int(color_rgba[1]*255)},{int(color_rgba[2]*255)},{alp2})'
                    colors_list.append(rgba_str)
                
                frame_data.append(
                    go.Scatter3d(
                        x=x_filtered, y=y_filtered, z=z_filtered,
                        mode='markers',
                        marker=dict(size=6, color=colors_list, line=dict(width=0)),
                        text=[f"E: {e:.2f} MeV" for e in E_filtered],
                        hovertemplate="<b>Voxel</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>%{text}<extra></extra>",
                        showlegend=False
                    )
                )
                
                frame_data.append(
                    go.Scatter3d(
                        x=[15, 15], y=[15, 15], z=[0, min(z_layer, 30)],
                        mode='lines',
                        line=dict(color='black', width=4),
                        hoverinfo='skip',
                        showlegend=False
                    )
                )
            
            z_frames.append(go.Frame(data=frame_data, name=f"z_{int(z_layer):02d}"))
    
    elif sort_by == "energy":
        # Energy-progression: show hits from highest to lowest energy
        all_hits = []
        for mat in materials:
            data = event_dict[mat]
            for x, y, z, e in zip(data['x'], data['y'], data['z'], data['E']):
                all_hits.append({'x': x, 'y': y, 'z': z, 'E': e, 'mat': mat})
        
        all_hits_sorted = sorted(all_hits, key=lambda h: h['E'], reverse=True)
        total_hits = len(all_hits_sorted)
        step_size = max(1, total_hits // 50)
        
        for num_hits in range(0, total_hits + 1, step_size):
            frame_data = []
            for mat in materials:
                data = event_dict[mat]
                energy_min = np.min(data['E'])
                energy_max = np.max(data['E'])
                norm = mpl.colors.LogNorm(vmin=energy_min, vmax=energy_max)
                
                mat_hits = [h for h in all_hits_sorted[:num_hits] if h['mat'] == mat]
                
                if len(mat_hits) == 0:
                    frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', showlegend=False))
                    frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', showlegend=False))
                    continue
                
                x_hits = np.array([h['x'] for h in mat_hits])
                y_hits = np.array([h['y'] for h in mat_hits])
                z_hits = np.array([h['z'] for h in mat_hits])
                E_hits = np.array([h['E'] for h in mat_hits])
                
                colors_list = []
                for e in E_hits:
                    norm_val = norm(e)
                    color_rgba = cmap(norm_val)
                    alp2 = 0.1 + 0.9 * np.log(e * 10) / np.log(energy_max * 10)
                    alp2 = float(np.clip(alp2, 0.1, 0.95))
                    rgba_str = f'rgba({int(color_rgba[0]*255)},{int(color_rgba[1]*255)},{int(color_rgba[2]*255)},{alp2})'
                    colors_list.append(rgba_str)
                
                frame_data.append(
                    go.Scatter3d(
                        x=x_hits, y=y_hits, z=z_hits,
                        mode='markers',
                        marker=dict(size=6, color=colors_list, line=dict(width=0)),
                        text=[f"E: {e:.2f} MeV" for e in E_hits],
                        hovertemplate="<b>Voxel</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>%{text}<extra></extra>",
                        showlegend=False
                    )
                )
                
                frame_data.append(
                    go.Scatter3d(
                        x=[15, 15], y=[15, 15], z=[0, 30],
                        mode='lines',
                        line=dict(color='black', width=4),
                        hoverinfo='skip',
                        showlegend=False
                    )
                )
            
            progress = num_hits / max(total_hits, 1)
            energy_frames.append(go.Frame(data=frame_data, name=f"e_{progress:.2f}"))
    
    # ===== CREATE INITIAL FRAME (all data) =====
    initial_traces = []
    for idx, mat in enumerate(materials):
        data = event_dict[mat]
        energy_min = np.min(data['E'])
        energy_max = np.max(data['E'])
        norm = mpl.colors.LogNorm(vmin=energy_min, vmax=energy_max)
        
        colors_list = []
        for e in data['E']:
            norm_val = norm(e)
            color_rgba = cmap(norm_val)
            alp2 = 0.1 + 0.9 * np.log(e * 10) / np.log(energy_max * 10)
            alp2 = float(np.clip(alp2, 0.1, 0.95))
            rgba_str = f'rgba({int(color_rgba[0]*255)},{int(color_rgba[1]*255)},{int(color_rgba[2]*255)},{alp2})'
            colors_list.append(rgba_str)
        
        initial_traces.append(
            go.Scatter3d(
                x=data['x'], y=data['y'], z=data['z'],
                mode='markers',
                marker=dict(size=6, color=colors_list, line=dict(width=0)),
                text=[f"E: {e:.2f} MeV" for e in data['E']],
                hovertemplate="<b>Voxel</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>%{text}<extra></extra>",
                showlegend=False
            )
        )
        
        initial_traces.append(
            go.Scatter3d(
                x=[15, 15], y=[15, 15], z=[0, 30],
                mode='lines',
                line=dict(color='black', width=4),
                hoverinfo='skip',
                showlegend=False
            )
        )
    
    for trace in initial_traces:
        fig.add_trace(trace)
    
    # ===== SELECT CORRECT FRAMES =====
    if sort_by == "z":
        fig.frames = z_frames
    else:
        fig.frames = energy_frames
        # Calculate total hits for energy mode
        all_hits = []
        for mat in materials:
            data = event_dict[mat]
            for x, y, z, e in zip(data['x'], data['y'], data['z'], data['E']):
                all_hits.append({'x': x, 'y': y, 'z': z, 'E': e, 'mat': mat})
        total_hits = len(all_hits)
    
    slider_steps = [
        dict(
            args=[[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
            label=f.name.split('_')[1] if sort_by == "z" else f"{int(float(f.name.split('_')[1]) * total_hits)}",  # ✅ Convert to hit count
            method='animate'
        )
        for f in fig.frames
    ]
    
    # Set up layout with animation controls
    layout_updates = {
        'title_text': f"Interactive Event Viewer - {sort_by.upper()} Progression",
        'height': 800 * rows,
        'width': 1000 * cols,
        'showlegend': False,
        'updatemenus': [
            dict(
                type='buttons',
                showactive=True,
                y=0.95, x=0.01, xanchor='left', yanchor='top',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}])
                ]
            )
        ],
        'sliders': [
            dict(
                active=0,
                yanchor='top', y=-0.02,
                xanchor='left', x=0.05,
                len=0.2,
                transition={'duration': 0},
                steps=slider_steps
            )
        ],
        'margin': dict(l=10, r=10, t=100, b=50)
    }
    
    for i in range(1, num_mats + 1):
        scene_name = 'scene' if i == 1 else f'scene{i}'
        layout_updates[scene_name] = dict(
            camera=camera,
            xaxis=dict(title="x [cells]", title_font=dict(size=20), tickfont=dict(size=14), range=[0, 30]),
            yaxis=dict(title="y [cells]", title_font=dict(size=20), tickfont=dict(size=14), range=[0, 30]),
            zaxis=dict(title="z [layers]", title_font=dict(size=20), tickfont=dict(size=14), range=[0, 30]),
            aspectmode='cube'
        )
    
    fig.update_layout(**layout_updates)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"event_viewer_{sort_by}_{materials[0]}.html")
    fig.write_html(output_file)
    print(f"Event viewer saved to: {output_file}")
    
    return output_file   

