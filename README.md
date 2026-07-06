# IR Spectra Pipeline

A Python pipeline for parsing NWChem and xTB DFT output files into IR spectroscopic data for use in machine learning workflows. Built as part of the SIMCODES REU program at Iowa State University.

---

## Project Structure

```
IR_SPEC_pipeline/
├── data/
│   ├── *.out                             # Example NWChem/xTB output files
│   └── *_results.csv                     # Parsed output CSVs
├── parsers/
│   ├── NWchem_parser_V2.py               # Parses NWChem .out files -> CSV
│   ├── xtb_parser_edited.py              # Parses xTB .out files -> CSV
│   └── Seq2Smile_parser.py               # Parses seq2mSMILES summary.tsv files -> CSV
├── modules/
│   ├── SMILESappend.py                   # Merges SMILES values into nwchem_results.csv
│   └── broaden_spectrum.py               # Gaussian-broadens a parsed spectrum CSV
├── data_exploration/
│   └── Convergence_check.py              # Reports % of converged geometries in a CSV
├── matplot/
│   └── data_plot.py                      # Plots each molecule's IR spectrum to NWChem_Plots/
├── requirements.txt
└── README.md
```

---

## What It Does

- Parses NWChem and xTB frequency-calculation output files (`.out`)
- Extracts total DFT energy, geometry convergence status, vibrational frequencies (cm⁻¹), and IR intensities (KM/mol)
- Bins IR intensities into a fixed-width wavenumber grid (50–4000 cm⁻¹, step 5) suitable for ML model input
- Incrementally updates the output CSV — files already present (by filename) are skipped on subsequent runs
- Extracts SMILES strings and coordination numbers from seq2mSMILES `summary.tsv` files and can merge them into parsed results
- Applies optional Gaussian broadening to binned spectra so linear compression (SVD/PCA) has structure to exploit
- Plots each parsed molecule's IR spectrum to PNG for quick visual inspection

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

## File Naming Convention

Both the NWChem and xTB parsers derive `amino_acid`, `metal`, `confirmation`, `basis`, and `theory` from the filename, so `.out` files must follow this 5-field, underscore-separated convention:

```
AA_Metal_Confirmation_Basis_Theory.out
```

Examples currently in `data/`: `AA_Cu_C1_B1_T1.out`, `AA_Co+2_C4_0_xtb.out`.

---

## How to Run

**Step 0** — Run your NWChem (`task dft optimize` + `task dft freq`) or xTB frequency calculations.

**Step 1 — Parse NWChem output:**
```bash
python parsers/NWchem_parser_V2.py \
    --folder data \
    --output data/nwchem_results.csv
```

**Step 1 (alt) — Parse xTB output:**
```bash
python parsers/xtb_parser_edited.py \
    --folder data \
    --output data/xtb_results.csv
```
Both default to `--folder data` if omitted; xTB additionally skips any filename that doesn't split into 5 underscore-separated fields.

**Step 2 (optional) — Parse SMILES/coordination data from seq2mSMILES output:**
```bash
python parsers/Seq2Smile_parser.py --input AH_CO+2 --output data/smiles_results.csv
```
Walks `--input` recursively for `summary.tsv` files and extracts `msmiles` and `coordination_number` per amino acid/metal pair.

**Step 3 (optional) — Apply Gaussian broadening to a parsed spectrum CSV:**
```bash
python modules/broaden_spectrum.py --input data/nwchem_results.csv --output data/nwchem_results_broadened.csv
```

**Step 4 (optional) — Plot parsed spectra:**
```bash
python matplot/data_plot.py
```
Reads `data/nwchem_results.csv` and writes one PNG per molecule to `NWChem_Plots/`, skipping any plot that already exists.

**Step 5 (optional) — Check convergence rate of a parsed CSV:**
```bash
python data_exploration/Convergence_check.py
```
Currently hardcoded to read `data/xtb_results_cleantest.csv`; edit the path to point at another parsed CSV.

> `modules/SMILESappend.py` is a one-off script that merges SMILES values from `data/fake_test_results.csv` into `data/nwchem_results.csv` — it has hardcoded paths and is intended as a reference for writing similar merge steps rather than a general-purpose CLI tool.

---

## Output Format

Both parsers produce the same schema:

| filename | amino_acid | metal | mol_name | confirmation | basis | theory | smiles | energy | converged | spec_50 | spec_55 | ... | spec_4000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AA_Cu_C1_B1_T1.out | AA | Cu | AA_Cu | C1 | B1 | T1 | NaN | -284.42 | True | 0.0 | 0.0 | ... | 1.785 |

---

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn
