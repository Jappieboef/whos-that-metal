# IR_Spec_2026 — ML Identification of Metals from IR Spectra

**SIMCODES REU 2026 · Iowa State University**
Author: Jasper Steenwinkel · Mentor: Dr. Theresa Windus

## Overview

This project identifies which metal (Co²⁺, Cu²⁺, Ni²⁺, or Zn²⁺) is bound in an
amino-acid–metal complex directly from its IR spectrum, using machine learning,
as a non-destructive alternative to mass spectrometry. Non-destructive
identification would allow rare-earth / transition metals to be measured and
then recovered or recycled.

The repository covers two halves of the project:
1. A **data pipeline** that turns raw quantum-chemistry output (NWChem, xTB)
   into fixed-grid, ML-ready IR spectra.
2. A **learning-curve study** quantifying how much training data a previously
   unseen metal needs before a classifier can reliably recognize it.

> **Note:** the actual dataset files (e.g. `vibspectrum_broadened.csv`) are not
> version-controlled — only `data/.gitkeep` is tracked, so the `data/` folder
> exists as a placeholder. Populate it by running the parsing/broadening
> pipeline below, or by dropping in a data archive separately.

---

## Repository structure

Only files tracked by git are listed below. Anything not shown here (e.g. a
local `venv/`, cached `__pycache__/`, or scratch scripts) is gitignored —
legacy code and dead-end experiments have been excluded.

```
IR_SPEC_pipeline/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── .gitkeep                          # placeholder — real spectra CSVs are not tracked
│
├── parsers/                              # raw quantum-chem output → structured data
│   ├── NWchem_parser_V2.py               # parses NWChem DFT output
│   ├── Seq2Smile_parser.py               # parses seq2mSMILES summaries
│   ├── vibspectrum_parser.py             # parses xTB `vibspectrum` files
│   └── xtb_parser_edited.py              # parses general xTB `.out` output
│
├── modules/                              # shared processing utilities
│   ├── broaden_spectrum.py               # Gaussian broadening onto fixed grid
│   └── SMILESappend.py                   # appends/joins SMILES data into the dataset
│
├── data_exploration/                     # exploratory / diagnostic scripts
│   ├── Convergence_check.py              # checks xTB/NWChem convergence status
│   └── pca_metal_separation.py           # PCA exploration of metal separability
│
├── Machine_Learning_Model/                # classification models & experiments
│   ├── lda_metal_separation.py           # Linear Discriminant Analysis baseline
│   ├── random_forest_metal.py            # Random Forest classifier
│   └── newcomer_dosing_curve_singleconf_5fold_freshsplits.py
│                                          # canonical learning-curve experiment:
│                                          # doses a "newcomer" metal 5%→100% into a
│                                          # model trained on the other 3 metals,
│                                          # single-conformer, 5-fold cross-validated
│
├── matplot/                              # figure-generation scripts
│   ├── aminoacid_overview.py             # amino-acid composition overview plot
│   ├── data_plot.py                      # general dataset plotting utilities
│   ├── spectra_overview.py               # overview plot of IR spectra by metal
│   └── plot_newcomer_dosing_curve.py     # polished plotter for the dosing-curve
│                                          # results (merged legend, origin-anchored,
│                                          # Okabe-Ito palette, 300 DPI + vector output)
│
├── Visuals/                               # generated result figures
│   ├── gb_confusion.png                  # Gradient Boosting confusion matrix
│   ├── lda_confusion_matrix.png          # LDA confusion matrix
│   ├── pca_metal_separation.png          # PCA projection of metal complexes
│   ├── spectra_overview_vibspectrum.png  # IR spectra overview by metal
│   └── newcomer_dosing_curve_holdout.png # learning-curve result figure
│
├── NWChem_Plots/
│   └── AA_Cu_C1_B1_T1.out.png            # rendered NWChem output for one AA_Cu complex
│
├── Notebook/
│   └── demo_pipeline_walkthrough.ipynb   # synthetic-data walkthrough of pipeline mechanics
│
├── AA_Co+2_C10_0_xtb/                     # <!-- inferred: sample raw xTB run output for one
│                                          # AA+Co+2 conformer (10th conformer, basis/theory 0);
│                                          # confirm if this is meant as a reference example -->
│   ├── AA_Co+2_C10_0_xtb.out
│   ├── AA_Co+2_C10_0_xtb.xyz
│   ├── charges
│   ├── g98.out
│   ├── hessian
│   ├── run_cmd.sh
│   ├── vibspectrum
│   ├── wbo
│   ├── xtbopt.log
│   ├── xtbopt.xyz
│   ├── xtbrestart
│   ├── xtbtopo.mol
│   └── .xtboptok
│
└── AH_CO+2/                               # <!-- inferred: donor-candidate analysis output
                                           # for the AH+Co+2 complex; confirm scope -->
    ├── all_donor_candidates.tsv
    ├── msmiles_output.txt
    ├── selected_donors.tsv
    ├── summary.tsv
    └── top_hypotheses.tsv
```

---

## Pipeline

1. **Parse raw quantum-chemistry output** (`parsers/`) — NWChem DFT logs, xTB
   `.out` files, and xTB `vibspectrum` files are parsed into structured
   records. Metadata (amino acid, metal, conformer, basis, theory) is derived
   from the `AA_Metal_Conf_Basis_Theory` filename/directory convention.
2. **Broaden and bin spectra** (`modules/broaden_spectrum.py`) — raw
   frequency/intensity pairs are converted to a fixed 50–4000 cm⁻¹ grid
   (step 5, 791 bins) via Gaussian broadening.
3. **Join into an ML-ready dataset** (`modules/SMILESappend.py` and the
   parsers above) — produces one row per molecule with metadata columns plus
   `spec_50 … spec_4000`.
4. **Explore** (`data_exploration/`) — convergence checks and PCA separability
   analysis.
5. **Classify** (`Machine_Learning_Model/`) — LDA and Random Forest baselines,
   plus the canonical newcomer-dosing learning-curve experiment.
6. **Visualize** (`matplot/`, outputs in `Visuals/`) — spectra overviews,
   confusion matrices, PCA plots, and the polished learning-curve figure.

---

## Results (headline numbers)

<!-- Filled in from prior session notes — confirm these still match the
     tracked single-conformer, 5-fold script before publishing. -->

- Gradient Boosting is the strongest classifier tried, well above chance
  (0.25) and the LDA baseline (0.66).
- The newcomer-dosing learning curve shows Zn²⁺ and Cu²⁺ are recognized with
  a smaller data budget than Co²⁺ and Ni²⁺, which are harder to separate from
  each other.
- Feature importance concentrates in three coordination-sensitive IR regions:
  metal–ligand stretch (~300–600 cm⁻¹), carboxylate/amide (~1500 cm⁻¹), and
  O–H/N–H stretch (~3600 cm⁻¹).

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Data files are not included in the repository — populate `data/` by running
the pipeline against raw xTB/NWChem output, or by pulling a prepared data
archive separately.