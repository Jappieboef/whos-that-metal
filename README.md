# IR Spectra Pipeline

A Python pipeline for parsing quantum chemistry output files (NWChem DFT and xTB) and extracting IR spectroscopic data for use in machine learning workflows. Built as part of the SIMCODES REU program at Iowa State University.

---

## Project Structure

```
IR_SPEC_pipeline/
├── data/                                   # input .out files + parsed output CSVs (CSVs are gitignored)
│   ├── AA_Cu_C1_B1_T1.out                  # example NWChem output file
│   └── AA_Co+2_C4_0_xtb.out                # example xTB output file
├── AH_CO+2/                                # example seq2mSMILES output (summary.tsv, msmiles_output.txt, ...)
├── NWChem_Plots/                           # IR spectrum plots, written by matplot/data_plot.py
├── parsers/
│   ├── NWchem_parser_V2.py                 # parses NWChem DFT frequency .out files
│   ├── xtb_parser_edited.py                # parses xTB frequency .out files
│   └── Seq2Smile_parser.py                 # parses seq2mSMILES summary.tsv files
├── modules/
│   ├── broaden_spectrum.py                 # Gaussian-broadens binned spectra (SVD/PCA preprocessing)
│   └── SMILESappend.py                     # fills in missing SMILES on an existing results CSV
├── matplot/
│   └── data_plot.py                        # plots intensity vs. wavenumber for every parsed row
├── data_exploration/
│   └── Convergence_check.py                # quick QC script: % of rows with converged == True
├── requirements.txt
└── README.md
```

---

## What It Does

- Parses **NWChem** DFT frequency-calculation output files (`.out`): total DFT energy, geometry convergence, vibrational frequencies (cm⁻¹), and IR intensities (KM/mol)
- Parses **xTB** frequency-calculation output files (`.out`) with the same schema, for faster semi-empirical results alongside full DFT
- Bins IR intensities into a fixed-width wavenumber grid (50–4000 cm⁻¹, step 5) suitable for ML model input, from either quantum chemistry method
- Parses **seq2mSMILES** `summary.tsv` output to recover SMILES strings and coordination numbers, and can back-fill missing `smiles` values onto an existing parsed results CSV
- Applies Gaussian broadening to binned spectra as a preprocessing step for linear dimensionality reduction (SVD/PCA), since raw binned peaks land in a single bin with no correlation to their neighbors
- Plots the IR spectrum (intensity vs. wavenumber) for every parsed row and saves it as a PNG
- Incrementally updates output CSVs — already-processed files are skipped by filename on subsequent runs
- Supports multiple basis-set/method subfolders under `data/`, tracking files by relative path to avoid duplicate processing

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

---

## How to Run

**Step 0** — Run your NWChem or xTB frequency calculations. NWChem jobs must include both `task dft optimize` and `task dft freq` to produce the IR data this pipeline needs.

**Step 1** — Place your `.out` files inside `data/` (or any folder you point `--folder` at). You can organize them into subfolders by basis set/method.

***FILES MUST FOLLOW THIS NAMING CONVENTION: `AminoAcid_Metal_Confirmation_Basis_Theory.out`***, e.g. `AA_Cu_C1_B1_T1.out` for NWChem, or `AA_Co+2_C4_0_xtb.out` for xTB (xTB runs use `0` as a basis placeholder since it's basis-set-free, and `xtb` as the theory field).

```
data/
├── 6-31G/
│   └── AA_Cu_C1_B1_T1.out
└── cc-pVDZ/
    └── AA_Cu_C1_B2_T1.out
```

**Step 2** — Run the appropriate parser:

```bash
# NWChem
python parsers/NWchem_parser_V2.py \
    --folder data \                        # default: data
    --output data/nwchem_results.csv       # default: data/nwchem_results.csv

# xTB
python parsers/xtb_parser_edited.py \
    --folder data \                        # default is an HPC cluster path — override for local runs
    --output data/xtb_results.csv          # default is an HPC cluster path — override for local runs
```

**Step 3 (optional)** — Recover SMILES strings from seq2mSMILES output, and back-fill them onto a parsed results CSV:

```bash
python parsers/Seq2Smile_parser.py --input AH_CO+2 --output data/smiles_results.csv
python modules/SMILESappend.py   # reads/overwrites data/nwchem_results.csv in place; requires data/fake_test_results.csv
```

**Step 4 (optional)** — Gaussian-broaden the binned spectra before feeding them into SVD/PCA or another compression step:

```bash
python modules/broaden_spectrum.py \
    --input data/nwchem_results.csv \
    --output data/nwchem_results_broadened.csv \
    --step 5 \
    --fwhm 20.0
```

**Step 5 (optional)** — Plot every parsed row's IR spectrum:

```bash
python matplot/data_plot.py   # reads data/nwchem_results.csv, writes PNGs to NWChem_Plots/
```

**Output** — Each parser writes one row per molecule to its output CSV, with columns for energy, convergence, and IR intensity at each wavenumber bin (`spec_50`, `spec_55`, ..., `spec_4000`), along with metadata about the sample.

---

## Output Format

| filename | amino_acid | metal | confirmation | basis | theory | smiles | energy | converged | spec_50 | spec_55 | ... | spec_4000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AA_Cu_C1_B1_T1.out | AA | Cu | C1 | B1 | T1 | | -284.42 | True | 0.0 | 0.0 | ... | 1.785 |
| AA_Co+2_C4_0_xtb.out | AA | Co+2 | C4 | 0 | xtb | | -59.65 | True | 53.96 | 0.0 | ... | 0.0 |

---

## Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn (reserved for upcoming ML work — not yet imported anywhere)
