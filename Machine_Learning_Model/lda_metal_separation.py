"""
Linear Discriminant Analysis (LDA) for metal separation.

PCA (unsupervised) showed only a faint metal gradient, because metal identity is
a LOW-VARIANCE feature: the dominant variance in the spectra comes from the amino
acid and the conformation. LDA is supervised: it actively searches for the
directions that best SEPARATE the four metals. This script deliberately shows BOTH:

  1. The naive in-sample LDA projection   -> looks as separated as LDA can make it
  2. A GROUP-AWARE cross-validated score  -> whether that separation is real
     or just LDA memorizing (the honest number).

The gap between the pretty in-sample plot and the CV accuracy is the whole point.
LDA plateaus because the true boundary between metals is NONLINEAR -- which is
why the production model is gradient boosting, not a linear classifier.

Figure styling matches the rest of the project: Okabe-Ito metal palette,
bold black spines/ticks, superscripted charges, and no per-figure metal legend
(a single global legend asset is used across the poster).
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score

INPUT = "data/vibspectrum_broadened.csv"
FIGURES_DIR = "figures"

# superscripts render upright in the body font (so charges look chemical, not italic)
plt.rcParams.update({"mathtext.default": "regular", "mathtext.fontset": "dejavusans"})

# ---- Categorical palette (metals): Okabe-Ito, locked project-wide ----
# Okabe & Ito (2008), "Color Universal Design" -- distinguishable under the most
# common forms of color blindness. Identical mapping in every project figure.
METAL_COLORS = {
    "Co+2": "#0072B2",  # blue
    "Cu+2": "#009E73",  # bluish green
    "Ni+2": "#E69F00",  # orange
    "Zn+2": "#D55E00",  # vermillion
}

# ---- Sequential palette (confusion matrix): ColorBrewer "Blues" ----
# Truncated slightly off the near-white low end so mid-range cells stay legible,
# while a true 0 still reads as (almost) blank.
CM_CMAP = LinearSegmentedColormap.from_list(
    "BluesCM", plt.get_cmap("Blues")(np.linspace(0.02, 0.95, 256)))

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# ---- shared figure styling ----
SPINE_COLOR = "black"
SPINE_WIDTH = 2.5
TICK_FONTSIZE = 18
TICK_LENGTH = 8
TICK_WIDTH = 2
TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 17
ANNOT_FONTSIZE = 15

# Scatter: with ~10k points, small + semi-transparent keeps the cluster shape
# readable instead of a solid blob.
POINT_SIZE = 12
POINT_ALPHA = 0.45


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


def load_data(path):
    df = pd.read_csv(path)
    before = len(df)
    df = df[(df["converged"] == True) & (df["freq_inten_mismatch"] == False)].reset_index(drop=True)
    print(f"Loaded {before:,} rows -> {len(df):,} after converged & non-mismatch filter.")

    spec_cols = [c for c in df.columns if c.startswith("spec_")]
    finite = np.isfinite(df[spec_cols].to_numpy(dtype=float)).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        print(f"Dropping {n_bad} row(s) with non-finite spectral values.")
        df = df[finite].reset_index(drop=True)

    # rows missing a label or group key can't train or score (and would poison
    # the group array with NaN)
    n_before = len(df)
    df = df.dropna(subset=["metal", "amino_acid"]).reset_index(drop=True)
    if len(df) != n_before:
        print(f"Dropping {n_before - len(df)} row(s) missing metal/amino_acid.")
    return df


def get_matrix(df):
    spec_cols = sorted((c for c in df.columns if c.startswith("spec_")),
                       key=lambda c: int(c.split("_")[1]))
    X = df[spec_cols].to_numpy(dtype=float)
    return X


def normalize_features(X):
    """Same recipe as the PCA script: per-spectrum L2, drop low-variance bins,
    per-feature standardize, guard against non-finite."""
    X = normalize(X, norm="l2", axis=1)
    std = X.std(axis=0)
    keep = std > max(1e-8, 1e-4 * std.max())
    X = X[:, keep]
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def grouping_key(df):
    """All conformations of the same amino_acid+metal must stay in the same CV
    fold, or the score is inflated by near-duplicate spectra leaking across folds."""
    return (df["amino_acid"].astype(str) + "_" + df["metal"].astype(str)).to_numpy()


def style_axes(ax, grid=True):
    """Bold black spines + long, thick, large-labelled ticks (project standard)."""
    ax.set_facecolor(SURFACE)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major",
                   length=TICK_LENGTH, width=TICK_WIDTH,
                   labelsize=TICK_FONTSIZE, color="black",
                   labelcolor=INK_PRIMARY)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_visible(True)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_data(INPUT)
    X = normalize_features(get_matrix(df))
    y = df["metal"].to_numpy()
    groups = grouping_key(df)

    print(f"\nClasses: {sorted(set(y))}")
    print(f"Samples: {len(y)}   Groups (AA+metal): {len(set(groups))}")

    # ---------------------------------------------------------------- projection
    lda = LinearDiscriminantAnalysis(n_components=2)
    proj = lda.fit_transform(X, y)          # fit and transform on ALL data
    insample_acc = accuracy_score(y, lda.predict(X))
    print(f"\nIn-sample LDA accuracy (optimistic, likely inflated): {insample_acc:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    # plot in fixed order so layering is reproducible, never data-dependent
    for label in METAL_COLORS:
        m = y == label
        if not m.any():
            continue
        ax.scatter(proj[m, 0], proj[m, 1], s=POINT_SIZE, alpha=POINT_ALPHA,
                   color=METAL_COLORS[label], linewidths=0, zorder=3)
        # label each cluster at its centroid instead of a legend box
        cx, cy = proj[m, 0].mean(), proj[m, 1].mean()
        ax.text(cx, cy, format_metal(label), fontsize=ANNOT_FONTSIZE + 3,
                fontweight="bold", ha="center", va="center",
                color=METAL_COLORS[label], zorder=6,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")])

    ax.set_title("LDA Projection (Supervised)",
                 color=INK_PRIMARY, loc="center",
                 fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)
    ax.set_xlabel("LD1", color=INK_SECONDARY,
                  fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("LD2", color=INK_SECONDARY,
                  fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "lda_metal_projection.png"), dpi=300,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/lda_metal_projection.png")

    # ---------------------------------------------------------------- honest CV
    pipe = Pipeline([("lda", LinearDiscriminantAnalysis())])
    cv = GroupKFold(n_splits=5)
    y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv)

    cv_acc = accuracy_score(y, y_pred)
    cv_bal = balanced_accuracy_score(y, y_pred)
    print(f"\nGroup-CV LDA accuracy (honest):          {cv_acc:.3f}")
    print(f"Group-CV LDA balanced accuracy:          {cv_bal:.3f}")
    print(f"Random-guess baseline (4 classes):       {1/len(set(y)):.3f}")

    # ---------------------------------------------------------------- confusion
    labels = sorted(set(y))
    disp = [format_metal(l) for l in labels]
    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(cm_norm, cmap=CM_CMAP, vmin=0, vmax=1)
    style_axes(ax, grid=False)

    ax.set_xticks(range(len(labels)), disp)
    ax.set_yticks(range(len(labels)), disp)
    ax.set_xlabel("Predicted metal", color=INK_SECONDARY,
                  fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("True metal", color=INK_SECONDARY,
                  fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(f"Cross-validated confusion matrix (row-normalized)",
                 color=INK_PRIMARY, loc="center",
                 fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else INK_PRIMARY,
                    fontsize=ANNOT_FONTSIZE,
                    fontweight="bold" if i == j else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="black", labelcolor=INK_PRIMARY,
                        labelsize=TICK_FONTSIZE - 4,
                        length=TICK_LENGTH - 3, width=TICK_WIDTH)
    cbar.outline.set_edgecolor(SPINE_COLOR)
    cbar.outline.set_linewidth(SPINE_WIDTH)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "lda_confusion_matrix.png"), dpi=300,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/lda_confusion_matrix.png")


if __name__ == "__main__":
    main()