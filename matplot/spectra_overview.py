import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap



INPUT = "data/vibspectrum_broadened.csv" #change this based on your dataset
OUTPUT_DIR = "matplot" #change this for the name you want your file to have
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "spectra_overview_vibspectrum.png")

# color pallatte imported
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

METAL_COLORS = {
    "Co+2": "#2a78d6",  # slot 1 blue
    "Cu+2": "#1baf7a",  # slot 2 aqua
    "Ni+2": "#eda100",  # slot 3 yellow
    "Zn+2": "#008300",  # slot 4 green
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_spectra(path): #sets the dataframe from the loaded data
    df = pd.read_csv(path)
    df = df[df["converged"] == True].reset_index(drop=True)
    spec_cols = sorted(
        (c for c in df.columns if c.startswith("spec_")),
        key=lambda c: int(c.split("_")[1]),
    )
    x = np.array([int(c.split("_")[1]) for c in spec_cols])
    return df, spec_cols, x


def main():
    df, spec_cols, x = load_spectra(INPUT)
    df = df.sort_values("metal", kind="stable").reset_index(drop=True)
    intensities = df[spec_cols].to_numpy(dtype=float)

    # sqrt compresses the huge dynamic range (mostly zero, occasional sharp
    # peaks) so the heatmap isn't just a black field with a few bright dots.
    scaled = np.sqrt(intensities)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax_heat, ax_lines) = plt.subplots(
        2, 1, figsize=(11, 9), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.patch.set_facecolor(SURFACE)

    ax_heat.set_facecolor(SURFACE)
    im = ax_heat.imshow(
        scaled,
        aspect="auto",
        cmap=BLUE_CMAP,
        extent=[x.min(), x.max(), len(df), 0],
        interpolation="nearest",
    )
    ax_heat.set_title(
        f"IR spectra overview - {len(df):,} converged calculations, grouped by metal center",
        color=INK_PRIMARY, fontsize=13, loc="left", pad=10,
    )
    ax_heat.tick_params(colors=INK_MUTED)
    for spine in ax_heat.spines.values():
        spine.set_color(GRID)
    ax_heat.set_yticks([])

    cbar = fig.colorbar(im, ax=ax_heat, pad=0.02, fraction=0.03)
    cbar.set_label("Intensity (sqrt scale)", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)
    cbar.outline.set_edgecolor(GRID)

    # metal-group separator lines + labels along the left margin
    blended = mtransforms.blended_transform_factory(ax_heat.transAxes, ax_heat.transData)
    boundaries = df["metal"].ne(df["metal"].shift()).to_numpy().nonzero()[0]
    boundaries = np.append(boundaries, len(df))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        metal = df["metal"].iloc[start]
        mid = (start + end) / 2
        ax_heat.text(
            -0.03, mid, metal, transform=blended,
            ha="right", va="center", fontsize=10, fontweight="bold",
            color=METAL_COLORS.get(metal, INK_MUTED),
        )
        if start > 0:
            ax_heat.axhline(start, color=SURFACE, linewidth=1.5)

    ax_lines.set_facecolor(SURFACE)
    ax_lines.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for metal in METAL_COLORS:  # fixed order, never derived from the data
        group = df[df["metal"] == metal]
        if group.empty:
            continue
        vals = group[spec_cols].to_numpy(dtype=float)
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        color = METAL_COLORS[metal]
        ax_lines.plot(x, mean, color=color, linewidth=2, label=f"{metal} (n={len(group)})", zorder=3)
        ax_lines.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0, zorder=2)

    ax_lines.set_xlabel("Wavenumber (cm$^{-1}$)", color=INK_SECONDARY)
    ax_lines.set_ylabel("Mean intensity", color=INK_SECONDARY)
    ax_lines.tick_params(colors=INK_MUTED)
    for spine in ax_lines.spines.values():
        spine.set_color(GRID)
    ax_lines.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved overview plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
