"""
Normalize the parsed IR spectra, run PCA, and see whether the projection
separates the 4 metal centers (Co+2, Cu+2, Ni+2, Zn+2).

Key fix vs. the earlier version: spectra are L2-normalized PER SPECTRUM first
(so overall intensity differences between molecules don't dominate), and THEN
standardized per feature for PCA. This stops a handful of very-high-intensity
spectra from hijacking the principal components.

The expected result is a NEGATIVE one, and that's the point: PCA is unsupervised,
so it finds the directions of greatest variance -- which are driven by the amino
acid and the conformation, not the metal. Metal identity is a LOW-VARIANCE
feature, so it does not surface in PC1/PC2. That is why the project moves to a
supervised method (LDA) and then to nonlinear models (gradient boosting).

Figure styling matches the rest of the project: Okabe-Ito metal palette, bold
black spines/ticks, superscripted charges, and no per-figure metal legend
(a single global legend asset is used across the poster). The amino-acid
chemistry-class panel keeps its own legend, since no global one exists for it.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

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

# Chemistry classes use the REMAINING Okabe-Ito colors so the two panels never
# imply a false correspondence between a class and a metal.
CLASS_COLORS = {
    "Nonpolar": "#56B4E9",   # sky blue
    "Aromatic": "#CC79A7",   # reddish purple
    "Polar":    "#F0E442",   # yellow
    "Basic":    "#0072B2",   # blue
    "Acidic":   "#D55E00",   # vermillion
    "Other":    "#999999",   # neutral grey
}

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
LEGEND_FONTSIZE = 13
ANNOT_FONTSIZE = 18

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
    print(f"Loaded {before:,} rows -> {len(df):,} remain after filtering to converged & non-mismatched.")

    # Drop rows whose spectrum contains non-finite values (inf/nan from parsing).
    # These are the true source of the 'divide by zero in matmul' PCA failure:
    # a single inf in a row becomes nan under normalization and poisons PCA.
    spec_cols = [c for c in df.columns if c.startswith("spec_")]
    finite_mask = np.isfinite(df[spec_cols].to_numpy(dtype=float)).all(axis=1)
    n_bad = int((~finite_mask).sum())
    if n_bad:
        bad_names = df.loc[~finite_mask, "dirname"].tolist() if "dirname" in df.columns else []
        print(f"Dropping {n_bad} row(s) with non-finite spectral values (inf/nan): "
              f"{bad_names[:10]}{' ...' if len(bad_names) > 10 else ''}")
        df = df[finite_mask].reset_index(drop=True)

    # Rows with no metal label can't be colored or interpreted; modern pandas also
    # propagates NaN through string ops, so drop them explicitly.
    n_before = len(df)
    df = df.dropna(subset=["metal"]).reset_index(drop=True)
    if len(df) != n_before:
        print(f"Dropping {n_before - len(df)} row(s) missing a metal label.")
    return df


def get_spec_matrix(df):
    """Return the ordered spec_ columns as a float matrix, plus the column names."""
    spec_cols = sorted(
        (c for c in df.columns if c.startswith("spec_")),
        key=lambda c: int(c.split("_")[1]),
    )
    X = df[spec_cols].to_numpy(dtype=float)
    return X, spec_cols


def report_outliers(df, X, n=8):
    """Print the spectra with the largest total intensity BEFORE normalization.
    These are the points that distort a raw PCA; worth eyeballing to confirm
    they are real molecules and not parser artifacts (e.g. one giant fake peak)."""
    total = X.sum(axis=1)
    order = np.argsort(total)[::-1][:n]
    label_col = "dirname" if "dirname" in df.columns else ("filename" if "filename" in df.columns else None)
    print(f"\nTop {n} spectra by total intensity (candidate outliers):")
    for idx in order:
        who = df[label_col].iloc[idx] if label_col else f"row {idx}"
        peak = X[idx].max()
        print(f"  {str(who):<32}  total={total[idx]:10.1f}  max_bin={peak:8.1f}")


def normalize_for_pca(X):
    """Per-spectrum L2 normalization (shape, not magnitude), then per-feature
    standardization (what PCA assumes).

    Critical for sparse spectra: bins that are zero (or near-zero) for almost
    every spectrum have a tiny standard deviation. Dividing by that tiny std
    blows the value up to inf/nan, which then poisons the PCA (the divide-by-zero
    / overflow warnings in matmul). So we drop LOW-variance bins, not just exactly
    -zero ones, and use a scaler that won't divide by a near-zero scale.
    """
    # 1) per-spectrum: each row -> unit L2 norm.
    X_rows = normalize(X, norm="l2", axis=1)

    sample_norms = np.linalg.norm(X_rows[:3], axis=1)
    print(f"Row-norm check (should be ~1.0): {np.round(sample_norms, 4)}")

    # 2) drop bins whose variance is negligible across the dataset. A relative
    #    threshold catches the many sparse bins that are ~always zero.
    col_std = X_rows.std(axis=0)
    thresh = max(1e-8, 1e-4 * col_std.max())
    keep = col_std > thresh
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped} low-variance spectral bin(s) "
              f"(<{thresh:.2e}); {int(keep.sum())} bins kept.")
    X_rows = X_rows[:, keep]

    # 3) per-feature standardization. with_std divides by the column std; after
    #    the low-variance drop above no column std is near zero, so this is safe.
    X_std = StandardScaler().fit_transform(X_rows)

    # final guard: replace any residual non-finite values so PCA never sees inf/nan
    n_bad = int((~np.isfinite(X_std)).sum())
    if n_bad:
        print(f"Guard: replacing {n_bad} non-finite value(s) with 0 before PCA.")
        X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)

    return X_std


def run_pca(X_std, n_components=5):
    pca = PCA(n_components=n_components, random_state=0)
    proj = pca.fit_transform(X_std)
    return pca, proj


def style_axes(ax):
    """Bold black spines + long, thick, large-labelled ticks (project standard)."""
    ax.set_facecolor(SURFACE)
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


def plot_projection(pca, proj, df):
    """Two-panel PCA scatter from the SAME projection: colored by metal, and by
    residue-1 chemistry class. The contrast is the actual result."""
    metal = df["metal"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2))
    fig.patch.set_facecolor(SURFACE)

    # ---- left: by metal (no legend; labelled at cluster centroids) ----
    style_axes(ax1)
    for label in METAL_COLORS:          # fixed order, never derived from the data
        mask = metal == label
        if not mask.any():
            continue
        color = METAL_COLORS[label]
        ax1.scatter(proj[mask, 0], proj[mask, 1], s=POINT_SIZE, alpha=POINT_ALPHA,
                    color=color, linewidths=0, zorder=3)
    # No in-plot metal labels: the clusters overlap heavily (that IS the PCA
    # result), so centroid text just collides. Identification comes from the
    # global metal legend asset.
    ax1.set_title("PCA Projection (Unsupervised)", color=INK_PRIMARY, loc="center",
                  fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)

    # ---- right: by amino-acid chemistry class (keeps its own legend) ----
    style_axes(ax2)
    aa_class = classify_residue(df)
    if aa_class is not None:
        for label, color in CLASS_COLORS.items():
            mask = aa_class == label
            if mask.any():
                ax2.scatter(proj[mask, 0], proj[mask, 1], s=POINT_SIZE,
                            alpha=POINT_ALPHA, color=color, label=label,
                            linewidths=0, zorder=3)
        ax2.set_title("Colored by amino acid (residue-1 chemistry class)",
                      color=INK_PRIMARY, loc="left",
                      fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)
        leg = ax2.legend(frameon=False, fontsize=LEGEND_FONTSIZE,
                         labelcolor=INK_SECONDARY, markerscale=2.2,
                         handletextpad=0.6, borderpad=0.4)
        for t in leg.get_texts():
            t.set_fontweight("bold")
    else:
        ax2.set_title("amino_acid column unavailable", color=INK_PRIMARY, loc="left",
                      fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)

    for ax in (ax1, ax2):
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
                      color=INK_SECONDARY, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
                      color=INK_SECONDARY, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")

    fig.tight_layout()
    outfile = os.path.join(FIGURES_DIR, "pca_metal_separation.png")
    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {outfile}")


def classify_residue(df):
    """Map the first residue's amino acid to a broad chemistry class.
    Returns None if amino_acid isn't usable (e.g. still a placeholder)."""
    if "amino_acid" not in df.columns:
        return None
    aa = df["amino_acid"].astype(str).str.upper().str.strip()
    # if it's all a placeholder like 'AA', there's no real residue identity yet
    if aa.nunique() <= 1:
        print("NOTE: amino_acid has <=1 unique value (placeholder?) - skipping class coloring.")
        return None

    groups = {
        "Nonpolar": {"GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "MET", "G", "A", "V", "L", "I", "P", "M"},
        "Aromatic": {"PHE", "TRP", "TYR", "F", "W", "Y"},
        "Polar": {"SER", "THR", "CYS", "ASN", "GLN", "S", "T", "C", "N", "Q"},
        "Basic": {"LYS", "ARG", "HIS", "K", "R", "H"},
        "Acidic": {"ASP", "GLU", "D", "E"},
    }
    lookup = {}
    for cls, members in groups.items():
        for m in members:
            lookup[m] = cls

    # Residue-1 identity. Names may be:
    #   - three-letter or full ('GLY', 'GLY-ALA')  -> take token before a delimiter
    #   - concatenated single-letter dipeptides ('LR', 'RY') -> take first CHARACTER
    def first_residue(name):
        for delim in ("-", "_", " "):
            if delim in name:
                return name.split(delim)[0]
        # no delimiter: 3-letter code maps whole; otherwise first char is residue 1
        if name in lookup:
            return name
        return name[0] if name else name

    first = aa.map(first_residue)
    mapped = first.map(lookup).fillna("Other")
    n_other = int((mapped == "Other").sum())
    if n_other:
        print(f"NOTE: {n_other} rows didn't map to a chemistry class (labeled 'Other').")
    return mapped.to_numpy()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_data(INPUT)
    X, _ = get_spec_matrix(df)

    report_outliers(df, X)

    X_std = normalize_for_pca(X)

    pca, proj = run_pca(X_std, n_components=5)
    print("\nExplained variance ratio (first 5 components):")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {ratio * 100:.2f}%")
    print(f"  First 2 components capture {pca.explained_variance_ratio_[:2].sum() * 100:.1f}% total.")

    plot_projection(pca, proj, df)


if __name__ == "__main__":
    main()