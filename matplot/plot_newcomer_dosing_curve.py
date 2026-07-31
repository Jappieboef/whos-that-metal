"""
plot_newcomer_dosing_curve_singleconf_5fold.py
------------------------------------------------
Reads newcomer_dosing_curve_singleconf_5fold_data.csv (from
newcomer_dosing_curve_singleconf_5fold.py) and renders a clean, poster-ready
figure — same polished style as plot_newcomer_dosing_curve.py, adapted for
the single-conformer / 5-fold CV run:

    - error bars = std of recall ACROSS THE 5 FOLDS at each dose, not
      across random model seeds, so the legend/annotations say so explicitly.
    - everything else (merged single legend, Okabe-Ito palette, origin
      anchor, 300 DPI PNG + vector PDF) is unchanged from the original
      plotter.

Expected columns (from newcomer_dosing_curve_singleconf_5fold.py):
    metal, dose_frac, n_newcomer_train, recall_mean, recall_std
Falls back gracefully if some are missing.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---- locked metal palette (Okabe-Ito) ----
COLORS = {
    "Co+2": "#0072B2",
    "Cu+2": "#009E73",
    "Ni+2": "#E69F00",
    "Zn+2": "#D55E00",
}
ORDER = ["Zn+2", "Cu+2", "Co+2", "Ni+2"]  # legend order = descending ceiling

ANCHOR_AT_ORIGIN = True  # draw each curve back to (0, 0): 0 samples -> 0 recall

# ---- IO ----
CANDIDATES = [
    "figures/newcomer_dosing_curve_singleconf_5fold_data.csv",
    "data/newcomer_dosing_curve_singleconf_5fold_data.csv",
    "newcomer_dosing_curve_singleconf_5fold_data.csv",
]
OUT_PNG = Path("figures/newcomer_dosing_curve_singleconf_5fold.png")

# ---- style ----
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
    "figure.dpi": 120,
    "axes.linewidth": 1.3,
    "svg.fonttype": "none",
    "mathtext.default": "regular",   # superscripts render upright in the body font
    "mathtext.fontset": "dejavusans",
})

# ---- tunable sizes / weights ----
SPINE_LW   = 2.6    # bold axis lines
TICK_LW    = 2.0
LABEL_SIZE = 17     # x / y axis labels
TITLE_SIZE = 18
SUBTITLE_SIZE = 12.5
TICK_SIZE  = 14
LEG_SIZE   = 12.5


def format_metal(m):
    """'Co+2' -> mathtext 'Co$^{2+}$' (charge as a superscript, chemistry order)."""
    mo = re.match(r"^\s*([A-Za-z]+)\s*([+-]?)\s*(\d*)\s*$", str(m))
    if not mo:
        return str(m)
    elem, sign, num = mo.groups()
    if not sign and not num:
        return elem
    charge = f"{num}{sign}" if num else sign      # "+2" -> "2+" (e.g. Co^{2+})
    return rf"{elem}$^{{{charge}}}$"


def find_csv():
    for c in CANDIDATES:
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "Could not find newcomer_dosing_curve_singleconf_5fold_data.csv "
        "in figures/, data/, or cwd."
    )


def main():
    df = pd.read_csv(find_csv())

    # column resolution (robust to minor naming differences)
    xcol = next((c for c in ["n_newcomer_train", "n_samples", "n"] if c in df.columns), None)
    if xcol is None:
        raise ValueError("No sample-count column (n_newcomer_train) found.")
    ycol = "recall_mean" if "recall_mean" in df.columns else "recall"
    ecol = "recall_std" if "recall_std" in df.columns else None

    metals = [m for m in ORDER if m in df["metal"].unique()]
    metals += [m for m in df["metal"].unique() if m not in metals]  # any extras

    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    handles, labels = [], []
    for metal in metals:
        sub = df[df["metal"] == metal].sort_values(xcol)
        x, y = sub[xcol].to_numpy(), sub[ycol].to_numpy()
        color = COLORS.get(metal, "#555555")

        # optionally anchor the curve at (0, 0): no samples -> no recall
        xl = np.insert(x, 0, 0.0) if ANCHOR_AT_ORIGIN else x
        yl = np.insert(y, 0, 0.0) if ANCHOR_AT_ORIGIN else y

        # line runs to the origin
        ax.plot(xl, yl, color=color, lw=2.4, zorder=3)

        # discrete error bars (with caps) at each real dose point -- no shaded band
        if ecol and ecol in sub:
            e = sub[ecol].to_numpy()
            ax.errorbar(x, y, yerr=e, fmt="o", ms=5, color=color,
                       markerfacecolor=color, markeredgecolor="white",
                       markeredgewidth=0.7, ecolor=color, elinewidth=1.6,
                       capsize=4, capthick=1.6, zorder=3)
        else:
            ax.plot(x, y, color=color, lw=0, zorder=3,
                    marker="o", ms=5, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=0.7)

        # one combined legend entry: color + ceiling (mean +/- 5-fold std) + n
        ceiling = y[-1]
        ceiling_std = sub[ecol].to_numpy()[-1] if ecol and ecol in sub else 0.0
        n_full = int(x[-1])
        handles.append(Line2D([0], [0], color=color, lw=2.6, marker="o",
                              markerfacecolor=color, markeredgecolor="white",
                              markeredgewidth=0.7, ms=6))
        labels.append(
            f"{format_metal(metal)}    {ceiling:.2f} \u00b1 {ceiling_std:.2f}    n = {n_full:,}"
        )

    # axes labels (larger + bold)
    ax.set_xlabel("New-metal training samples added",
                  fontsize=LABEL_SIZE, fontweight="bold", labelpad=9)
    ax.set_ylabel("Recall on held-out new metal",
                  fontsize=LABEL_SIZE, fontweight="bold", labelpad=9)
    ax.set_title("Introducing a new metal to a fully-trained 3-metal model",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=28)
    ax.text(0.5, 1.02, "1 conformer / molecule  \u00b7  error bars = std across 5-fold CV",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=SUBTITLE_SIZE, style="italic", color="#444444")

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, which="major", axis="both", alpha=0.25,
            linestyle="--", linewidth=0.7, zorder=0)

    # bold axis lines
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(SPINE_LW)
    ax.tick_params(labelsize=TICK_SIZE, width=TICK_LW, length=6)

    # single merged legend, parked in the empty lower-right triangle
    leg = ax.legend(
        handles, labels,
        title="New metal  \u00b7  recall @ 100% dose (mean \u00b1 5-fold std)",
        loc="lower right", bbox_to_anchor=(0.995, 0.02),
        fontsize=LEG_SIZE, title_fontsize=LEG_SIZE + 0.5,
        frameon=True, framealpha=0.95, borderpad=0.9,
        handletextpad=0.9, labelspacing=0.7,
    )
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_linewidth(1.0)
    leg.get_title().set_fontweight("bold")
    # left-align the multi-column text for a table feel
    for t in leg.get_texts():
        t.set_fontfamily("monospace")

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG.with_suffix(".pdf"), bbox_inches="tight")  # vector copy for the poster
    print(f"saved {OUT_PNG} and {OUT_PNG.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()