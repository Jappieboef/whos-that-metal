"""
Random Forest classifier for metal identification from IR spectra.

Scored honestly: GroupKFold grouped on amino_acid+metal so all conformations of
a complex stay on one side of the split (no leakage), reported as balanced
accuracy against the 0.25 random baseline and the 0.66 LDA reference.

Bonus: Random Forest gives feature importances, so we plot WHICH wavenumbers
drive the metal prediction - a chemistry result, not just an accuracy number.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

INPUT = "data/vibspectrum_broadened.csv"
FIGURES_DIR = "figures"
TARGET = "metal"          # switch to "amino_acid" to predict the other target
N_SPLITS = 5

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_data(path):
    df = pd.read_csv(path)
    print(df["amino_acid"].isna().sum(), "missing out of", len(df))
    print(df["amino_acid"].value_counts().head())
    df = df[df[TARGET].notna()].reset_index(drop=True)
    df = df[df[TARGET].astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"After dropping missing {TARGET}: {len(df):,} rows.")
    before = len(df)
    df = df[(df["converged"] == True) & (df["freq_inten_mismatch"] == False)].reset_index(drop=True)
    print(f"Loaded {before:,} -> {len(df):,} after converged & non-mismatch filter.")
    spec_cols = [c for c in df.columns if c.startswith("spec_")]
    finite = np.isfinite(df[spec_cols].to_numpy(dtype=float)).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        print(f"Dropping {n_bad} row(s) with non-finite spectral values.")
        df = df[finite].reset_index(drop=True)
    return df


def get_matrix(df):
    cols = sorted((c for c in df.columns if c.startswith("spec_")),
                  key=lambda c: int(c.split("_")[1]))
    wavenumbers = np.array([int(c.split("_")[1]) for c in cols])
    return df[cols].to_numpy(dtype=float), wavenumbers


def normalize_features(X):
    """Per-spectrum L2 -> drop low-variance bins -> standardize. Returns the
    scaled matrix AND the kept-column mask so importances map back to wavenumbers."""
    X = normalize(X, norm="l2", axis=1)
    std = X.std(axis=0)
    keep = std > max(1e-8, 1e-4 * std.max())
    X = X[:, keep]
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, keep


def grouping_key(df):
    return (df["amino_acid"].astype(str) + "_" + df["metal"].astype(str)).to_numpy()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_data(INPUT)
    X_raw, wavenumbers = get_matrix(df)
    X, keep = normalize_features(X_raw)
    wn_kept = wavenumbers[keep]
    y = df[TARGET].to_numpy()
    groups = grouping_key(df)
    labels = sorted(set(y))
    baseline = 1.0 / len(labels)

    print(f"\nTarget: {TARGET}   classes: {len(labels)}   samples: {len(y)}   "
          f"groups: {len(set(groups))}   features: {X.shape[1]}")

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    )

    # ---- honest cross-validated score ----
    cv = GroupKFold(n_splits=N_SPLITS)
    y_pred = cross_val_predict(rf, X, y, groups=groups, cv=cv)
    bal = balanced_accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    print(f"\nRandom Forest (group-CV):")
    print(f"  balanced accuracy: {bal:.3f}")
    print(f"  raw accuracy:      {acc:.3f}")
    print(f"  random baseline:   {baseline:.3f}")
    print(f"  LDA reference:     0.660")

    # ---- confusion matrix ----
    cm = confusion_matrix(y, y_pred, labels=labels).astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, color=INK_SECONDARY, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels, color=INK_SECONDARY)
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("True", color=INK_SECONDARY)
    ax.set_title("Random Forest - CV confusion (row-normalized)", color=INK_PRIMARY, loc="left", fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else INK_PRIMARY, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "rf_confusion.png"), dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"\nSaved {FIGURES_DIR}/rf_confusion.png")

    rf.fit(X, y)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    print("\nTop 10 most metal-informative wavenumbers:")
    for idx in order[:10]:
        print(f"  {wn_kept[idx]:>4} cm^-1   importance={importances[idx]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(wn_kept, importances, width=6, color="#378ADD")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", color=INK_SECONDARY)
    ax.set_ylabel("RF importance", color=INK_SECONDARY)
    ax.set_title(f"Which IR bands drive {TARGET} prediction", color=INK_PRIMARY, loc="left")
    ax.tick_params(colors=INK_SECONDARY)
    for s in ax.spines.values():
        s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "rf_importance.png"), dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/rf_importance.png")

if __name__ == "__main__":
    main()
