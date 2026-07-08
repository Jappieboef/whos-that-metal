"""
Normalize the parsed IR spectra, run PCA, and see whether the projection
separates the 4 metal centers (Co+2, Cu+2, Ni+2, Zn+2).

Key fix vs. the earlier version: spectra are L2-normalized PER SPECTRUM first
(so overall intensity differences between molecules don't dominate), and THEN
standardized per feature for PCA. This stops a handful of very-high-intensity
spectra from hijacking the principal components.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

INPUT = "data/vibspectrum_broadened.csv"
FIGURES_DIR = "figures"

# Fixed categorical order (not cycled), consistent with the other plots in this repo.
METAL_COLORS = {
    "Co+2": "#2a78d6",
    "Cu+2": "#1baf7a",
    "Ni+2": "#eda100",
    "Zn+2": "#008300",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


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
        print(f"  {who:<32}  total={total[idx]:10.1f}  max_bin={peak:8.1f}")


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


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_projection(pca, proj, df):
    """Two-panel PCA scatter from the SAME projection: colored by metal, and by
    residue-1 chemistry class. The contrast is the actual result."""
    metal = df["metal"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor(SURFACE)

    # ---- left: by metal ----
    _style_ax(ax1)
    for label, color in METAL_COLORS.items():
        mask = metal == label
        ax1.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.5, color=color,
                    label=label, linewidths=0)
    ax1.set_title("Colored by metal", color=INK_PRIMARY, loc="left")
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    # ---- right: by amino-acid chemistry class (if available) ----
    _style_ax(ax2)
    aa_class = classify_residue(df)
    if aa_class is not None:
        class_colors = {
            "Nonpolar": "#2a78d6", "Aromatic": "#1baf7a", "Polar": "#eda100",
            "Basic": "#6a51a3", "Acidic": "#e0555b", "Other": "#b8b6ae",
        }
        for label, color in class_colors.items():
            mask = aa_class == label
            if mask.any():
                ax2.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.5,
                            color=color, label=label, linewidths=0)
        ax2.set_title("Colored by amino acid (residue-1 chemistry class)",
                      color=INK_PRIMARY, loc="left")
        ax2.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    else:
        ax2.set_title("amino_acid column unavailable", color=INK_PRIMARY, loc="left")

    for ax in (ax1, ax2):
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", color=INK_SECONDARY)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", color=INK_SECONDARY)

    fig.tight_layout()
    outfile = os.path.join(FIGURES_DIR, "pca_metal_separation.png")
    fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor=SURFACE)
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