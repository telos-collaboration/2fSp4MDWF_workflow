#!/usr/bin/env python3
"""Compare Shamir plaquettes versus mass on 8^4 and 32 x 16^3 (no YM)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from bulk_phase import build_meta_map, is_true, parse_info


BETAS = (6.8, 7.0, 7.2, 7.4, 7.6)
LARGE_MASSES = (0.02, 0.06, 0.10)


def collect_points(paths, metadata):
    points = {}
    for path in paths:
        kind, key, info = parse_info(path)
        if kind != "dyn" or key not in metadata:
            raise ValueError(f"Input is not a metadata-listed dynamical ensemble: {path}")
        row = metadata[key]
        if not all(np.isclose(float(info[name]), value) for name, value in
                   [("NF", 2), ("Ls", 8), ("alpha", 1), ("a5", 1), ("M5", 1.8), ("mpv", 1)]):
            raise ValueError(f"Input does not have the requested Shamir parameters: {path}")
        beta, mass = float(info["beta"]), float(info["mass"])
        volume = (int(info["Nt"]), int(info["Ns"]))
        if beta not in BETAS:
            raise ValueError(f"Input beta is not in the shared selection: {path}")
        if volume == (8, 8):
            if not is_true(row.get("use_in_bulkphase_Shamir", False)):
                raise ValueError(f"Input is not a flagged Shamir bulk ensemble: {path}")
        elif volume != (32, 16) or mass not in LARGE_MASSES:
            raise ValueError(f"Unexpected volume or mass: {path}")
        point_key = (volume, beta, mass)
        if point_key in points:
            raise ValueError(f"Duplicate ensemble: {path}")
        with open(path) as handle:
            extract = json.load(handle)["hmc_extract"]
        value, error = float(extract["plaq"]), float(extract["plaq_err"])
        if not np.isfinite(value) or not np.isfinite(error) or error < 0:
            raise ValueError(f"Invalid plaquette average or uncertainty: {path}")
        points[point_key] = (value, error)

    expected = {
        (volume, beta, mass)
        for volume, masses in [((8, 8), [i / 100 for i in range(1, 11)]),
                               ((32, 16), LARGE_MASSES)]
        for beta in BETAS for mass in masses
    }
    if set(points) != expected:
        raise ValueError(f"Expected exactly 65 ensembles; missing {expected - set(points)}, "
                         f"unexpected {set(points) - expected}")
    return points


def plot_comparison(points, output, show_legend):
    # Match bulk_phase.py: viridis_r beta colors, error bars and paper styling.
    colors = plt.cm.viridis_r(np.linspace(0.1, 1.0, len(BETAS)))
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    for beta, color in zip(BETAS, colors):
        for volume, marker, linestyle in [((32, 16), "o", "-"), ((8, 8), "s", ":")]:
            values = sorted((mass, *result) for (vol, b, mass), result in points.items()
                            if vol == volume and b == beta)
            mass, plaq, error = np.asarray(values).T
            ax.errorbar(mass, plaq, yerr=error, color=color, marker=marker,
                        ls=linestyle, fillstyle="full",
                        markerfacecolor=color if volume == (32, 16) else "none")
    ax.set_xlabel(r"$am_0$")
    ax.set_ylabel(r"$\langle \mathcal{P} \rangle$")
    if show_legend:
        # Two explicit legend blocks keep beta colors separate from volumes.
        beta_legend = ax.legend(
            handles=[Line2D([], [], color=c, label=rf"$\beta={b:.1f}$")
                     for b, c in zip(BETAS, colors)],
            loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")
        ax.add_artist(beta_legend)
        ax.legend(handles=[
            Line2D([], [], color="black", marker="o", fillstyle="full", ls="-",
                   label=r"$V=32\times16^3$"),
            Line2D([], [], color="black", marker="s", markerfacecolor="none", ls=":",
                   label=r"$V=8^4$ (bulk scan)"),
        ], loc="lower left", bbox_to_anchor=(1.02, 0), fontsize="small")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensembles_csv", required=True)
    parser.add_argument("--plaq_avg", nargs="+", required=True)
    parser.add_argument("--label", default="yes")
    parser.add_argument("--plot_styles", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plt.style.use(args.plot_styles or "tableau-colorblind10")
    points = collect_points(args.plaq_avg, build_meta_map(args.ensembles_csv))
    plot_comparison(points, args.output, args.label.strip().lower() == "yes")
    print("Plotted 65 Shamir ensembles: 50 at V=8^4 and 15 at V=32x16^3; no YM.")


if __name__ == "__main__":
    main()
