# IR Spectra Pipeline — Metal Identification from Vibrational Spectra

**SIMCODES REU · Iowa State University**
**Mentor:** Dr. Theresa Windus · **Author:** Jasper Steenwinkel (GitHub: `Jappieboef`)

A Python pipeline that parses NWChem and xTB DFT frequency-calculation output
into ML-ready IR spectroscopic data, and trains classifiers that identify
**which metal is bound in an amino-acid–metal complex directly from its
vibrational spectrum** — a non-destructive alternative to mass spectrometry
(which consumes the sample), with applications in rare-earth / transition-metal
recovery and recycling.

**Project title:** *How Much Data Does a New Metal Need? A Learning-Curve
Study for ML Identification of Metal Complexes from IR Spectra*

---

## Status

| | |
|---|---|
| **Data pipeline** | ✅ Complete — deployed on Iowa State's Nova HPC via SLURM |
| **Dataset** | 10,900 converged xTB spectra · 4 metals (Co+2, Cu+2, Ni+2, Zn+2) · balanced (~2,700 each) |
| **Model selection** | ✅ Complete — **Gradient Boosting** chosen: 0.894 balanced accuracy (chance = 0.25, LDA = 0.66, RF = 0.855) |
| **Confusion pattern** | Cu (0.93) and Zn (0.95) clean; **Co↔Ni confused** (Co→Ni 13%, Ni→Co 9–11%) — reflects genuinely similar first-row coordination chemistry, not a fixable modeling gap (see [Experiments Explored](#experiments-explored-not-adopted)) |
| **Learning curve** | ✅ Headline result complete — new-metal (Cu) recall rises 0 → ~0.92, elbow ≈ 800–900 samples, sharp diminishing returns after |
| **Poster** | In progress — swapping in real figures |

---

## Project Structure

```
IR_SPEC_pipeline/
├── data/
│   ├── *.out                             # Example NWChem/xTB output files
│   ├── *_results.csv                     # Parsed output CSVs
│   └── vibspectrum_broadened.csv         # Final ML-ready joined dataset (main modeling input)
├── parsers/
│   ├── NWchem_parser_V2.py               # Parses NWChem .out files -> CSV
│   ├── xtb_parser_edited.py              # Parses xTB .out files -> CSV
│   ├── vibspectrum_parser.py             # Parses xTB vibspectrum files (metadata from parent dir name)
│   └── Seq2Smile_parser.py               # Parses seq2mSMILES summary.tsv files -> CSV
├── modules/
│   ├── SMILESappend.py                   # Merges SMILES values into nwchem_results.csv (reference only)
│   └── broaden_spectrum.py               # Gaussian-broadens a parsed spectrum CSV
├── data_exploration/
│   └── Convergence_check.py              # Reports % of converged geometries in a CSV
├── matplot/
│   └── data_plot.py                      # Plots each molecule's IR spectrum to NWChem_Plots/
├── perdiction_models/                               
│   ├── model_comparison.py               # Chance / LDA / RF / GB side-by-side comparison
│   ├── lda_metal_separation.py           # Linear baseline (0.66)
│   ├── pca_metal_separation.py           # Unsupervised exploration (see PCA vs LDA framing below)
│   ├── random_forest_metal.py            # RF runner-up (0.855)
│   ├── gradient_boosting_metal.py        # Chosen model (0.894) — confusion matrix + permutation importance
│   ├── rf_groupings.py                   # Grouping/exploration utilities
│   ├── learning_curve.py                 # HEADLINE experiment — Pattern A data-budget study
│   ├── hierarchical_co_ni_classifier.py  # Explored, not adopted (see below)
│   └── tune_gb_hyperparams.py            # Explored, inconclusive (see below)
├── outputs/
│   ├── figures/                          # gb_confusion.png, gb_importance.png, learning_curve.png, etc.
├── requirements.txt
└── README.md
```

---

## What It Does

**Parsing & feature engineering:**
- Parses NWChem and xTB frequency-calculation output files (`.out`), plus xTB `vibspectrum` files
- Extracts total DFT energy, geometry convergence status, vibrational frequencies (cm⁻¹), and IR intensities (KM/mol)
- Bins IR intensities into a fixed-width wavenumber grid (**50–4000 cm⁻¹, step 5 → 791 bins**), suitable for ML model input
- Incrementally updates the output CSV — files already present (by filename) are skipped on subsequent runs
- Extracts SMILES strings and coordination numbers from seq2mSMILES `summary.tsv` files, mergeable into parsed results
- Applies Gaussian broadening to binned spectra so linear compression (SVD/PCA) has structure to exploit
- Plots each parsed molecule's IR spectrum to PNG for quick visual inspection

**Modeling:**
- Classifies **metal** (4 classes: Co+2, Cu+2, Ni+2, Zn+2) directly from the 791-bin spectrum
- Compares Chance / LDA / Random Forest / Gradient Boosting; Gradient Boosting is the chosen model
- Runs a **learning-curve study** quantifying how many training spectra a previously-unseen metal needs before it's reliably recognized — the core scientific contribution of the project
- Also supports an **amino_acid** target (dipeptides, ~100+ classes) as a secondary, not-yet-completed comparison

---

## Data & File Conventions

- **Main modeling input:** `data/vibspectrum_broadened.csv` — one row per molecule. Columns: `amino_acid`, `metal`, `confirmation`, `basis`, `theory`, `smiles` (may be NaN), `energy`, `converged`, `freq_inten_mismatch`, then `spec_50 … spec_4000` (791 bins, step 5).
- **Filename / directory convention:** `AA_Metal_Confirmation_Basis_Theory` (5 underscore-separated fields), e.g. `AH_Co+2_C5_0_xtb`. For `vibspectrum` files, metadata comes from the **parent directory name**, not the filename itself.
- **Metal notation — watch for inconsistency:** older/example files in `data/` use bare metal symbols (`AA_Cu_C1_B1_T1.out`), but the current convention **and the ML dataset carry charge** (`Co+2`, `Cu+2`, `Ni+2`, `Zn+2`). Be consistent when joining datasets from different eras — a bare `Cu` and a `Cu+2` will not join or group correctly.
- **Always filter before modeling:** `converged == True` AND `freq_inten_mismatch == False`. Also drop rows with non-finite spectral values (see Known Issues).
- **Nova HPC paths:** parser + SLURM scripts under `/work/simcodes/IRSpectra/`; data under `.../XTB/`; parsed CSVs under `.../Parsed_Data/`. venv at `/work/simcodes/IRSpectra/venv/` — call its Python by full path in SLURM scripts (`source activate` is unreliable in batch jobs).

---

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/SIMCODES-ISU/IR_Spec_2026.git
cd IR_Spec_2026
```

**2. Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

> On Nova, SLURM batch jobs should call the venv's Python by full path
> (`/work/simcodes/IRSpectra/venv/bin/python`) rather than relying on
> `source activate`, which is unreliable in batch context.

---

## How to Run — Parsing Pipeline

**Step 0** — Run your NWChem (`task dft optimize` + `task dft freq`) or xTB frequency calculations.

**Step 1 — Parse NWChem output:**
```bash
python parsers/NWchem_parser_V2.py --folder data --output data/nwchem_results.csv
```

**Step 1 (alt) — Parse xTB output:**
```bash
python parsers/xtb_parser_edited.py --folder data --output data/xtb_results.csv
```
Both default to `--folder data` if omitted; xTB additionally skips any filename that doesn't split into 5 underscore-separated fields.

**Step 2 (optional) — Parse SMILES/coordination data from seq2mSMILES output:**
```bash
python parsers/Seq2Smile_parser.py --input AH_CO+2 --output data/smiles_results.csv
```
Walks `--input` recursively for `summary.tsv` files and extracts `msmiles` and `coordination_number` per amino acid/metal pair.

**Step 3 (optional) — Apply Gaussian broadening:**
```bash
python modules/broaden_spectrum.py --input data/nwchem_results.csv --output data/nwchem_results_broadened.csv
```

**Step 4 (optional) — Plot parsed spectra:**
```bash
python matplot/data_plot.py
```
Reads `data/nwchem_results.csv` and writes one PNG per molecule to `NWChem_Plots/`, skipping any plot that already exists.

**Step 5 (optional) — Check convergence rate:**
```bash
python data_exploration/Convergence_check.py
```
Currently hardcoded to read `data/xtb_results_cleantest.csv`; edit the path to point at another parsed CSV.

> `modules/SMILESappend.py` is a one-off script that merges SMILES values
> from `data/fake_test_results.csv` into `data/nwchem_results.csv` — it has
> hardcoded paths and is intended as a reference for writing similar merge
> steps, not a general-purpose CLI tool.

---

## How to Run — Modeling Pipeline

All modeling scripts read `data/vibspectrum_broadened.csv` by default (edit
the `INPUT` constant at the top of each script if your path differs) and
write figures to `figures/` or `outputs/figures/`.

**models:**
```bash
python models/lda_metal_separation.py
python models/random_forest_metal.py
python models/gradient_boosting_metal.py   # chosen model — writes gb_confusion.png, gb_importance.png
```

**Learning curve (the headline experiment):**
```bash
python models/learning_curve.py
```
Knobs at the top of the file: `N_REPEATS` (default 5), `DOSE_STEP` (default
0.05), `NEW_METAL` (which metal is held out as the "newcomer"). Drop to
`N_REPEATS=3`, `DOSE_STEP=0.10` for a fast pass; restore both for the final
smooth plot. Prints a live progress bar + ETA — full runs are slow (retrains
GB ~21 doses × `N_REPEATS` times).

**Experimental / exploratory (see results below before running):**
```bash
python models/hierarchical_co_ni_classifier.py --data data/vibspectrum_broadened.csv
python models/tune_gb_hyperparams.py --data data/vibspectrum_broadened.csv --n_iter 40
```

---

## Key Modeling Decisions (settled — do not re-litigate)

- **Task = classification, spectrum → metal.** Not regression.
- **Two possible targets:** `metal` (4 classes, main focus) and `amino_acid` (dipeptides, ~100+ classes, secondary/incomplete).
- **Chosen model = Gradient Boosting** (`HistGradientBoostingClassifier`). RF is the documented runner-up (0.855); LDA is the linear baseline (0.66).
- **Normalization recipe, in this exact order, fit on TRAIN only:** per-spectrum L2 → drop low-variance bins → per-feature StandardScaler → guard non-finite.
- **Grouping / no-leak rule:** group by `amino_acid + metal`. All conformations of one complex must stay on the same side of any split. Use `GroupKFold`. **This is the single most important correctness rule** — a random split inflates scores via conformation leakage.
- **Learning curve = "Pattern A":** frozen 20% test set for the new metal, carved once; dose the remaining 80% pool into training in 5% batches of whole complexes; evaluate every dose on the same frozen test set. X-axis is "% of pool," not "% of all data."
- **PCA vs LDA framing — say it precisely:** PCA underwhelms because metal is a *low-variance* component (PCA is unsupervised, chases dominant variation = ligand shape). LDA underwhelms at 0.66 because the metal signal is *nonlinear*. A supervised nonlinear model (gradient boosting) was needed. **Do not say "PCA failed because nonlinear"** — that conflates two separate reasons.

---

## Results Summary

| Model | Balanced Accuracy |
|---|---|
| Chance | 0.25 |
| LDA (linear) | 0.66 |
| Random Forest | 0.855 |
| **Gradient Boosting (chosen)** | **0.894** |

**Confusion matrix (GB, group-CV, row-normalized):**

| True \ Pred | Co+2 | Cu+2 | Ni+2 | Zn+2 |
|---|---|---|---|---|
| Co+2 | 0.83 | 0.03 | 0.13 | 0.01 |
| Cu+2 | 0.02 | 0.93 | 0.02 | 0.03 |
| Ni+2 | 0.10–0.11 | 0.03 | 0.86 | 0.01 |
| Zn+2 | 0.01 | 0.02 | 0.02 | 0.95 |

**Feature importances** concentrate in 3 coordination-sensitive regions: metal–ligand stretch (300–600 cm⁻¹), carboxylate/amide (~1500 cm⁻¹), O–H/N–H stretch (~3600 cm⁻¹) — "the model learned real chemistry." One open check: confirm the ~3600 cm⁻¹ importance isn't a boundary artifact (spectra go flat past ~3600 cm⁻¹); GB uses all 3 regions jointly so it's probably fine, but worth a sentence in the writeup.

**Learning curve (Pattern A, Cu as new metal):** recall rises 0 → ~0.92, elbow around n ≈ 800–900 Cu samples; most performance bought by the first ~400–900 samples, sharp diminishing returns after. Overall balanced accuracy and Cu recall converge around 60% dose. **Rotating the held-out metal across all 4** is the planned next step — the spread across 4 curves ("depending on the metal, you need between X and Y samples") is a stronger result than one curve alone.

---

## Experiments Explored (not adopted)

Documenting these because a controlled negative result is itself evidence,
and it's worth not re-running them expecting a different answer.

**Hierarchical two-stage classifier** (`hierarchical_co_ni_classifier.py`) —
Stage 1: 3-class GB (`Co_or_Ni` / Cu+2 / Zn+2); Stage 2: binary specialist
trained only on Co/Ni rows. Hypothesis: isolating the hard boundary into
a dedicated model would sharpen it. **Result: flat to slightly worse across
every class** (balanced accuracy 0.894 → 0.889; Co→Ni leak 13.1%→14.3%;
Ni→Co leak 10.1%→11.5%). Likely explanation: the joint 4-class model uses
Cu/Zn rows as implicit contrastive examples that indirectly sharpen the
Co/Ni boundary too; splitting the problem removed that context without
adding anything a tree ensemble couldn't already find via its own splits.
**Conclusion:** the Co↔Ni confusion is a genuine spectral-similarity limit
(consistent with their similar first-row coordination chemistry), not an
architecture problem.

**Hyperparameter tuning** (`tune_gb_hyperparams.py`) — GroupKFold-respecting
randomized search over `learning_rate`, `max_iter`, `max_depth`,
`max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`. In isolation the
search reported 0.906 (+0.0125 over untuned defaults). **This did not
reproduce** when the winning params were dropped into
`gradient_boosting_metal.py` (confusion-matrix diagonal averaged back to
~0.895) — likely because `HistGradientBoostingClassifier`'s default
`early_stopping='auto'` only activates above 10,000 training samples per
fold, so the search (which used defaults) effectively ran with early
stopping **off**, while the production script forces `early_stopping=True`
regardless of fold size, cutting the tuned model short of the iteration
count it was tuned for. **Not resolved** — the two scripts' early-stopping
behavior needs to be reconciled before this result can be trusted either
way. **Currently shipping the original untuned defaults (0.894)** as the
reported model.

---

## Known Issues / Tech Debt

1. **Parser writes occasional non-finite (inf/nan) rows.** Downstream scripts drop them at load time, so it's not blocking, but the root cause is upstream in the parser (likely a near-zero-frequency mode or a divide in broadening). Worth tracing eventually. ~39 rows also had blank `amino_acid` — trivial, dropped at load.
2. **Learning curve is slow on full data** — retrains GB ~21 doses × `N_REPEATS` times. See knobs above.
3. **`amino_acid` task not yet run to completion** — ~100+ classes, balanced ~40 each. Different problem than metal; compare to its own baseline, not to metal's 0.894.
4. **Grid edge check** — confirm the ~3600 cm⁻¹ importance isn't a boundary artifact.
5. **Hyperparameter search vs. production `early_stopping` mismatch** — see Experiments Explored above; needs reconciling if tuning is revisited.

---

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

(Optional, for the demo walkthrough notebook: `jupyter`, `nbformat`, `ipykernel`.)