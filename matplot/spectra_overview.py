import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.colors import PowerNorm, LinearSegmentedColormap


INPUT = "data/vibspectrum_broadened.csv" #change this based on your dataset
OUTPUT_DIR = "matplot" #change this for the name you want your file to have
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "spectra_overview_vibspectrum.png")

# superscripts render upright in the body font (so charges look chemical, not italic)
plt.rcParams.update({"mathtext.default": "regular", "mathtext.fontset": "dejavusans"})

# ---- Sequential palette (heatmap): matplotlib's built-in "Blues" ----
# The low end of "Blues" is near-white, which makes a sparse spectrum look washed
# out. We keep the palette but start partway in and run to the darkest end, so
# real peaks read as deep navy instead of pale blue.
CMAP_LO, CMAP_HI = 0.12, 1.0          # slice of "Blues" to use
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "BluesDark", plt.get_cmap("Blues")(np.linspace(CMAP_LO, CMAP_HI, 256)))

# ---- Heatmap contrast controls ----
# ROW_NORMALIZE: scale every spectrum by its own maximum, so one very intense
#   calculation can't wash out all the others (this is the main fix for a pale
#   heatmap on the real dataset).
# CLIP_PCT: values at/above this percentile saturate to the darkest color.
# GAMMA: <1 lifts mid-tones (more of the peak structure goes dark); 1.0 = linear.
ROW_NORMALIZE = True
CLIP_PCT = 99.0
GAMMA = 0.55

# ---- Categorical palette (metal lines/labels): Okabe-Ito ----
METAL_COLORS = {
    "Co+2": "#0072B2",  # blue
    "Cu+2": "#009E73",  # bluish green
    "Ni+2": "#E69F00",  # orange
    "Zn+2": "#D55E00",  # vermillion
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Borders/spines: darker, and now bolder so the plot boundary reads at poster distance.
SPINE_COLOR = "black"
SPINE_WIDTH = 2.5

# ---- font sizes (bumped up) ----
TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 15
TICK_FONTSIZE = 18
TICK_LENGTH = 8
TICK_WIDTH = 2
METAL_LABEL_FONTSIZE = 17

# ---- important spectral regions to highlight ----
# (lo, hi) in cm^-1, and a short label. The ~1500 and ~3600 features are given a
# +/-100 window so the single-wavenumber callouts read as bands.
# Regions are OUTLINED (black dashed box), not filled, so the underlying data
# stays fully visible.
BAND_EDGE = "black"
BAND_LW = 2
BAND_DASH = "dashed"
BANDS = [
    ((300, 600),   "metal-ligand"),
    ((1400, 1600), "carboxylate / amide"),
    ((3500, 3700), "O-H / N-H"),
]


def format_metal(m):
    """'Co+2' -> 'Co$^{2+}$' (charge as a superscript, chemistry order)."""
    mo = re.match(r"^\s*([A-Za-z]+)\s*([+-]?)\s*(\d*)\s*$", str(m))
    if not mo:
        return str(m)
    elem, sign, num = mo.groups()
    if not sign and not num:
        return elem
    charge = f"{num}{sign}" if num else sign      # "+2" -> "2+"
    return rf"{elem}$^{{{charge}}}$"


def load_spectra(path): #sets the dataframe from the loaded data
    df = pd.read_csv(path)
    df = df[df["converged"] == True].reset_index(drop=True)
    spec_cols = sorted(
        (c for c in df.columns if c.startswith("spec_")),
        key=lambda c: int(c.split("_")[1]),
    )
    x = np.array([int(c.split("_")[1]) for c in spec_cols])
    return df, spec_cols, x


def style_ticks(ax):
    """Match the demo: long, thick, black major ticks with large labels."""
    ax.tick_params(axis="both", which="major",
                   length=TICK_LENGTH, width=TICK_WIDTH,
                   labelsize=TICK_FONTSIZE, color="black",
                   labelcolor=INK_PRIMARY)


def style_spines(ax):
    """Apply the clearer, bolder spine style to every border of an axis."""
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_visible(True)


def outline_bands(ax, label=False, label_y=0.985, fontsize=11):
    """Draw a black dashed box around each important region (no fill)."""
    # x in data coords, y spans the full axis height
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for (lo, hi), name in BANDS:
        rect = mpatches.Rectangle(
            (lo, 0.0), hi - lo, 1.0, transform=trans,
            facecolor="none", edgecolor=BAND_EDGE, linewidth=BAND_LW,
            linestyle=BAND_DASH, zorder=6, clip_on=True,
        )
        ax.add_patch(rect)
        if label:
            txt = ax.text((lo + hi) / 2, label_y, name, transform=trans,
                          rotation=90, ha="center", va="top",
                          fontsize=fontsize, fontweight="bold",
                          color="#111111", zorder=7)
            txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])


def main():
    df, spec_cols, x = load_spectra(INPUT)
    df = df.sort_values("metal", kind="stable").reset_index(drop=True)
    intensities = df[spec_cols].to_numpy(dtype=float)

    # --- refine the heatmap dynamic range ---
    scaled = intensities.astype(float)
    if ROW_NORMALIZE:
        # each spectrum against its own peak -> every row uses the full color range
        rowmax = scaled.max(axis=1, keepdims=True)
        rowmax[rowmax <= 0] = 1.0
        scaled = scaled / rowmax
    # saturate the top percentile so faint structure isn't crushed by outliers
    vmax = np.percentile(scaled, CLIP_PCT)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(scaled.max()) or 1.0
    scaled = np.clip(scaled, 0.0, vmax)

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
        norm=PowerNorm(gamma=GAMMA, vmin=0.0, vmax=vmax),
        # thousands of rows get downsampled to screen pixels; averaging avoids the
        # speckled aliasing that "nearest" produces on the full dataset
        interpolation="antialiased",
        rasterized=True,
    )
    ax_heat.set_title(
        f"IR spectra overview - {len(df):,} converged calculations, grouped by metal center",
        color=INK_PRIMARY, fontsize=TITLE_FONTSIZE, fontweight="bold", loc="left", pad=10,
    )
    style_ticks(ax_heat)
    style_spines(ax_heat)
    ax_heat.set_yticks([])



    # outline the important regions on the heatmap (with vertical labels)
    outline_bands(ax_heat, label=True)

    # metal-group separator lines + labels along the left margin
    blended = mtransforms.blended_transform_factory(ax_heat.transAxes, ax_heat.transData)
    boundaries = df["metal"].ne(df["metal"].shift()).to_numpy().nonzero()[0]
    boundaries = np.append(boundaries, len(df))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        metal = df["metal"].iloc[start]
        mid = (start + end) / 2
        ax_heat.text(
            -0.04, mid, format_metal(metal), transform=blended,
            ha="right", va="center", fontsize=METAL_LABEL_FONTSIZE, fontweight="bold",
            color=METAL_COLORS.get(metal, INK_SECONDARY),
        )
        if start > 0:
            ax_heat.axhline(start, color=SURFACE, linewidth=1.5)

    ax_lines.set_facecolor(SURFACE)
    ax_lines.grid(True, color=GRID, linewidth=0.8, zorder=0)
    # outline the important regions on the mean-spectra panel
    outline_bands(ax_lines)
    for metal in METAL_COLORS:  # fixed order, never derived from the data
        group = df[df["metal"] == metal]
        if group.empty:
            continue
        vals = group[spec_cols].to_numpy(dtype=float)
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        color = METAL_COLORS[metal]
        ax_lines.plot(x, mean, color=color, linewidth=2, zorder=3)
        ax_lines.fill_between(x, mean - std, mean + std, color=color, alpha=0.15,
                              linewidth=0, zorder=2)

    ax_lines.set_xlabel("Wavenumber (cm$^{-1}$)", color=INK_SECONDARY,
                        fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_lines.set_ylabel("Mean intensity", color=INK_SECONDARY,
                        fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    style_ticks(ax_lines)
    style_spines(ax_lines)

    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved overview plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()