"""
newcomer_dosing_curve_singleconf_5fold_freshsplits.py
-------------------------------------------------------
Same experiment as newcomer_dosing_curve_singleconf_5fold.py (single
conformer per molecule, background metals always at 100%, dosed 5% -> 100%),
but changes how the error bars are computed.

THE PROBLEM THIS FIXES:
In the previous version, each metal's 5-fold split was drawn ONCE and then
reused at every dose along the curve. That means the test-fold composition
never changes as you move along the x-axis -- so the fold-to-fold spread
mostly reflects "which molecules happened to land in which fold" rather than
shrinking as more training data is added, which is the intuitive expectation
for error bars on a learning curve.

THE FIX:
At every dose, redraw a FRESH random 5-fold split of the newcomer's
molecules, and do this N_SPLIT_REPEATS times (independent re-assignments),
not just once. All (repeat x fold) recall values collected at that dose are
pooled together into recall_mean / recall_std. Consequences:
    - The test-fold composition is no longer fixed across the curve, so
      persistent "this fold has the hard molecules" effects get averaged
      away instead of showing up as a constant-width band all the way across.
    - At low doses, the dosed subset is also freshly resampled each repeat,
      which is an additional source of variance that fades out as dose ->
      100% (at 100% dose the whole pool is used every time, so that
      particular source of spread disappears; only fold-composition
      variance remains).
    - Cost: this multiplies the number of model fits by N_SPLIT_REPEATS
      compared to the single-split version. Drop N_SPLIT_REPEATS to 1 for a
      fast first pass (that degenerates to "fresh split every dose, no
      averaging"); use 3-5 for the final figure.

Outputs (into figures/):
    figures/newcomer_dosing_curve_singleconf_5fold_freshsplits.png
    figures/newcomer_dosing_curve_singleconf_5fold_freshsplits_data.csv
"""

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_PATH = "data/vibspectrum_broadened.csv"
FIG_DIR = Path("figures")
FIG_PATH = FIG_DIR / "newcomer_dosing_curve_singleconf_5fold_freshsplits.png"
CSV_PATH = FIG_DIR / "newcomer_dosing_curve_singleconf_5fold_freshsplits_data.csv"

METALS = ["Co+2", "Cu+2", "Ni+2", "Zn+2"]

DOSE_STEP = 0.05          # 5% increments
N_FOLDS = 5               # folds per split (still an 80/20 train/test split per fold)
N_SPLIT_REPEATS = 3       # independent fresh fold re-assignments PER DOSE, averaged
LOW_VAR_THRESHOLD = 1e-8  # drop near-constant bins
RANDOM_STATE = 42
CONFORMER_COL = "confirmation"  # per-molecule conformer id column, e.g. "C1", "C5"
CONFORMER_NUMBER = 1            # keep this conformer number for every molecule

# Okabe-Ito colorblind-safe palette (matches the other figures)
COLORS = {
    "Co+2": "#0072B2",
    "Cu+2": "#009E73",
    "Ni+2": "#E69F00",
    "Zn+2": "#D55E00",
}

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans", "sans-serif"]


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)

    spec_cols = [c for c in df.columns if c.startswith("spec_")]

    df = df[(df["converged"] == True) & (df["freq_inten_mismatch"] == False)].copy()

    spec = df[spec_cols].to_numpy(dtype=float)
    finite = np.isfinite(spec).all(axis=1)
    df = df.loc[finite].reset_index(drop=True)

    # group = amino_acid + metal -> one row per MOLECULE (all its conformers share this)
    df["group_id"] = df["amino_acid"].astype(str) + "_" + df["metal"].astype(str)

    n_before = len(df)
    df = select_one_conformer_per_molecule(df)
    print(f"Single-conformer filter: {n_before} rows -> {len(df)} rows "
          f"({df['group_id'].nunique()} unique molecules)")

    return df, spec_cols


def select_one_conformer_per_molecule(df, conformer_col=CONFORMER_COL,
                                       conformer_number=CONFORMER_NUMBER):
    """Keep only conformer number `conformer_number` (default 1, e.g. 'C1') for
    each (amino_acid, metal) molecule. No energy column available, so this is
    a fixed pick rather than a lowest-energy pick.
    """
    conf_num = df[conformer_col].astype(str).str.extract(r"(\d+)$")[0].astype(float)
    mask = conf_num == float(conformer_number)
    kept = df.loc[mask].copy()

    dupes = kept["group_id"].duplicated().sum()
    if dupes:
        print(f"WARNING: {dupes} molecules had more than one row matching "
              f"conformer {conformer_number} -- keeping the first per molecule.")
        kept = kept.drop_duplicates(subset="group_id", keep="first")

    return kept.reset_index(drop=True)


def normalize(X_train, X_test):
    """Per-spectrum L2 -> drop low-variance bins -> StandardScaler. Fit on TRAIN only."""
    def l2(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    X_train, X_test = l2(X_train), l2(X_test)

    keep = X_train.var(axis=0) > LOW_VAR_THRESHOLD
    X_train, X_test = X_train[:, keep], X_test[:, keep]

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train, X_test


def newcomer_recall(model, X_test, newcomer):
    """Test set is entirely the newcomer, so recall = fraction predicted as newcomer."""
    y_pred = model.predict(X_test)
    return float(np.mean(y_pred == newcomer))


def fmt(seconds):
    """Seconds -> compact h/m/s string."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ----------------------------------------------------------------------
# CORE
# ----------------------------------------------------------------------
def run_newcomer(df, spec_cols, newcomer, progress=None):
    """For each dose, redraw N_SPLIT_REPEATS fresh 5-fold splits of the
    newcomer's molecules (rather than reusing one fixed split across the
    whole curve). progress: dict with 'done'/'total'/'t0' for global ETA."""
    rng = np.random.default_rng(RANDOM_STATE)
    metal_t0 = time.perf_counter()

    is_new = df["metal"] == newcomer

    # Background: the other 3 metals, ALWAYS 100% (single conformer each, already filtered)
    bg = df.loc[~is_new]
    X_bg = bg[spec_cols].to_numpy(dtype=float)
    y_bg = bg["metal"].to_numpy()

    new_df = df.loc[is_new].reset_index(drop=True)
    n_mol = len(new_df)

    print(f"\n{newcomer} as newcomer")
    print(f"  background (3 metals, 100%): {len(bg):>6} molecules")
    print(f"  newcomer molecules:          {n_mol:>6}  -> {N_SPLIT_REPEATS} fresh "
          f"{N_FOLDS}-fold splits PER DOSE")

    doses = np.arange(DOSE_STEP, 1.0 + 1e-9, DOSE_STEP)

    rows = []
    for dose in doses:
        dose_t0 = time.perf_counter()
        dose_recalls = []
        dose_ns = []

        for split_rep in range(N_SPLIT_REPEATS):
            # fresh fold assignment for THIS dose and THIS repeat
            order = np.arange(n_mol)
            rng.shuffle(order)
            folds = np.array_split(order, N_FOLDS)

            for fold_i in range(N_FOLDS):
                test_idx = folds[fold_i]
                pool_idx = np.concatenate(
                    [folds[j] for j in range(N_FOLDS) if j != fold_i]
                )
                rng.shuffle(pool_idx)  # dosing order within this split's pool

                test_df = new_df.iloc[test_idx]
                X_test_raw = test_df[spec_cols].to_numpy(dtype=float)

                n_dose = max(1, int(round(dose * len(pool_idx))))
                dosed_idx = pool_idx[:n_dose]
                dosed = new_df.iloc[dosed_idx]

                X_new = dosed[spec_cols].to_numpy(dtype=float)
                y_new = dosed["metal"].to_numpy()

                X_train_raw = np.vstack([X_bg, X_new])
                y_train = np.concatenate([y_bg, y_new])

                X_train, X_test = normalize(X_train_raw, X_test_raw)

                model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
                model.fit(X_train, y_train)
                recall = newcomer_recall(model, X_test, newcomer)

                dose_recalls.append(recall)
                dose_ns.append(len(dosed))

                if progress is not None:
                    progress["done"] += 1

        dose_dt = time.perf_counter() - dose_t0

        rows.append(
            {
                "metal": newcomer,
                "dose_frac": round(float(dose), 2),
                "n_newcomer_train": int(round(np.mean(dose_ns))),
                "recall_mean": float(np.mean(dose_recalls)),
                "recall_std": float(np.std(dose_recalls)),
                "fit_seconds": round(dose_dt, 2),
            }
        )

        msg = (f"    dose {dose:4.0%}  n~{int(round(np.mean(dose_ns))):>5}  "
               f"recall = {np.mean(dose_recalls):.3f} \u00b1 {np.std(dose_recalls):.3f}  "
               f"[{fmt(dose_dt)}]")

        if progress is not None:
            elapsed = time.perf_counter() - progress["t0"]
            frac = progress["done"] / progress["total"]
            eta = elapsed / frac - elapsed if frac > 0 else 0.0
            msg += (f"  |  {progress['done']}/{progress['total']} fits "
                    f"({frac:.0%})  elapsed {fmt(elapsed)}  ETA {fmt(eta)}")

        print(msg, flush=True)

    print(f"  {newcomer} finished in {fmt(time.perf_counter() - metal_t0)}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------
def plot(results):
    fig, ax = plt.subplots(figsize=(9, 6))

    for metal in METALS:
        sub = results[results["metal"] == metal].sort_values("n_newcomer_train")
        x = sub["n_newcomer_train"].to_numpy()
        y = sub["recall_mean"].to_numpy()
        e = sub["recall_std"].to_numpy()

        ax.plot(x, y, color=COLORS[metal], lw=2.2, zorder=3, label=metal)
        ax.errorbar(x, y, yerr=e, fmt="o", ms=5, color=COLORS[metal],
                   markerfacecolor=COLORS[metal], markeredgecolor="white",
                   markeredgewidth=0.7, ecolor=COLORS[metal], elinewidth=1.6,
                   capsize=4, capthick=1.6, zorder=3)

    ax.set_xlabel("New-metal training samples added", fontsize=14, labelpad=10)
    ax.set_ylabel("Recall on held-out new metal", fontsize=14, labelpad=10)
    ax.set_title(
        "Introducing a new metal to a fully-trained 3-metal model\n"
        "(single conformer / molecule, fresh 5-fold split per dose)",
        fontsize=15, pad=14,
    )

    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(labelsize=12, width=1.2)

    ax.legend(title="New metal", fontsize=12, title_fontsize=12,
              frameon=False, loc="lower right")

    lines = []
    for metal in METALS:
        sub = results[results["metal"] == metal]
        top = sub.loc[sub["dose_frac"].idxmax()]
        lines.append(f"{metal}:  {top['recall_mean']:.2f} \u00b1 {top['recall_std']:.2f}  "
                     f"(n~{int(top['n_newcomer_train'])})")

    ax.text(
        0.98, 0.06,
        "Recall at 100% dose (mean \u00b1 std, fresh splits)\n" + "\n".join(lines),
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.95),
        zorder=4,
    )

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {FIG_PATH}")


# ----------------------------------------------------------------------
def main():
    t0 = time.perf_counter()

    df, spec_cols = load_data(DATA_PATH)
    print(f"Loaded {len(df)} molecules (1 conformer each) | "
          f"metals: {sorted(df['metal'].unique())}")

    n_doses = len(np.arange(DOSE_STEP, 1.0 + 1e-9, DOSE_STEP))
    total_fits = n_doses * N_SPLIT_REPEATS * N_FOLDS * len(METALS)
    print(f"Plan: {len(METALS)} metals x {n_doses} doses x {N_SPLIT_REPEATS} split-repeats "
          f"x {N_FOLDS} folds = {total_fits} fits\n")

    progress = {"done": 0, "total": total_fits, "t0": t0}

    results = pd.concat(
        [run_newcomer(df, spec_cols, m, progress) for m in METALS],
        ignore_index=True,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {CSV_PATH}")

    plot(results)

    print(f"\nTotal runtime: {fmt(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
