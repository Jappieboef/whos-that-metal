import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# The "amino_acid" column is a two-residue code (e.g. "AA", "AH", "RR") -
# a dipeptide, not a single category. With ~280 distinct pairs a legend or
# per-line color is useless, but the pair structure itself is a 20x20 grid:
# every cell is one (residue 1, residue 2) combination. This plots that grid
# directly instead of trying to force 280 categories into a line plot.

INPUT = "data/vibspectrum_broadened.csv"
OUTPUT_DIR = "matplot"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "aminoacid_overview_total.png")

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
BLUE_CMAP.set_bad("#e1e0d9")  # no-data cells: gridline gray, not part of the scale

# Second heatmap shown alongside the first gets the next categorical slot's
# hue (aqua) instead of repeating blue, so the two metrics read as distinct
# sequential scales rather than one continued scale.
AQUA_RAMP = ["#d7f5ea", "#a9e6cd", "#6dd1ab", "#1baf7a", "#128a5f", "#0a6647", "#053d2b"]
AQUA_CMAP = LinearSegmentedColormap.from_list("aqua_seq", AQUA_RAMP)
AQUA_CMAP.set_bad("#e1e0d9")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Residues grouped by side-chain chemistry (categorical slots 1-5, fixed order),
# then concatenated into one 20-long axis order so the grid reads as five
# chemistry blocks instead of a flat alphabet.
GROUPS = [
    ("Nonpolar",  "G A V L I P M", "#2a78d6"),  # slot 1 blue
    ("Aromatic",  "F Y W",         "#1baf7a"),  # slot 2 aqua
    ("Polar",     "S T C N Q",     "#eda100"),  # slot 3 yellow
    ("Basic",     "K R H",         "#4a3aa7"),  # slot 5 violet
    ("Acidic",    "D E",           "#e34948"),  # slot 6 red
]
ORDER = [r for _, letters, _ in GROUPS for r in letters.split()]
RESIDUE_COLOR = {r: color for _, letters, color in GROUPS for r in letters.split()}


def load_pairs(path):
    df = pd.read_csv(path)
    df = df[df["converged"] == True]
    df = df[df["amino_acid"].astype(str).str.len() == 2].copy()
    df["res1"] = df["amino_acid"].str[0]
    df["res2"] = df["amino_acid"].str[1]
    spec_cols = sorted(
        (c for c in df.columns if c.startswith("spec_")),
        key=lambda c: int(c.split("_")[1]),
    )
    x = np.array([int(c.split("_")[1]) for c in spec_cols])
    return df, spec_cols, x


def build_grids(df, spec_cols, x):
    n = len(ORDER)
    total = np.full((n, n), np.nan)
    peak_pos = np.full((n, n), np.nan)
    counts = np.zeros((n, n), dtype=int)

    grouped = df.groupby(["res1", "res2"])
    for (r1, r2), group in grouped:
        if r1 not in RESIDUE_COLOR or r2 not in RESIDUE_COLOR:
            continue
        i, j = ORDER.index(r1), ORDER.index(r2)
        mean_spec = group[spec_cols].to_numpy(dtype=float).mean(axis=0)
        total[i, j] = np.trapezoid(mean_spec, x)
        peak_pos[i, j] = x[np.argmax(mean_spec)]
        counts[i, j] = len(group)

    return total, peak_pos, counts


def style_grid_axis(ax):
    n = len(ORDER)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ORDER, fontsize=8)
    ax.set_yticklabels(ORDER, fontsize=8)
    for tick, r in zip(ax.get_xticklabels(), ORDER):
        tick.set_color(RESIDUE_COLOR[r])
    for tick, r in zip(ax.get_yticklabels(), ORDER):
        tick.set_color(RESIDUE_COLOR[r])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    # separators between chemistry blocks
    boundary = 0
    for _, letters, _ in GROUPS[:-1]:
        boundary += len(letters.split())
        ax.axhline(boundary - 0.5, color=SURFACE, linewidth=2)
        ax.axvline(boundary - 0.5, color=SURFACE, linewidth=2)


def main():
    df, spec_cols, x = load_pairs(INPUT)
    total, peak_pos, counts = build_grids(df, spec_cols, x)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax_total, ax_peak) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor(SURFACE)

    im1 = ax_total.imshow(np.ma.masked_invalid(total), cmap=BLUE_CMAP)
    ax_total.set_title(
        "Total IR activity by residue pair", color=INK_PRIMARY, fontsize=12, loc="left",
    )
    style_grid_axis(ax_total)
    cbar1 = fig.colorbar(im1, ax=ax_total, fraction=0.046, pad=0.04)
    cbar1.set_label("Integrated intensity", color=INK_SECONDARY)
    cbar1.ax.tick_params(colors=INK_MUTED)

    im2 = ax_peak.imshow(np.ma.masked_invalid(peak_pos), cmap=AQUA_CMAP)
    ax_peak.set_title(
        "Dominant peak location by residue pair", color=INK_PRIMARY, fontsize=12, loc="left",
    )
    style_grid_axis(ax_peak)
    cbar2 = fig.colorbar(im2, ax=ax_peak, fraction=0.046, pad=0.04)
    cbar2.set_label("Wavenumber (cm$^{-1}$)", color=INK_SECONDARY)
    cbar2.ax.tick_params(colors=INK_MUTED)

    for ax in (ax_total, ax_peak):
        ax.set_xlabel("Residue 2", color=INK_SECONDARY)
    ax_total.set_ylabel("Residue 1", color=INK_SECONDARY)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, markersize=8, label=name)
        for name, _, color in GROUPS
    ]
    handles.append(
        plt.Line2D([0], [0], marker="s", linestyle="", color=GRID, markersize=8, label="Not yet computed")
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles), frameon=False,
        fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.02),
    )

    n_pairs = int((counts > 0).sum())
    fig.suptitle(
        f"{n_pairs} of {len(ORDER)**2} possible residue-pair combinations computed "
        f"({df.shape[0]:,} converged spectra)",
        color=INK_SECONDARY, fontsize=10, y=0.98,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved amino acid overview to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
