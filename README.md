# IR Spectra Pipeline

A Python pipeline for parsing NWChem DFT output files and extracting IR spectroscopic data for use in machine learning workflows. Built as part of the SIMCODES REU program at Iowa State University.

---

## Project Structure

```
IR_SPEC_pipeline/
├── data/               # NWChem .out files and output CSV
├── matplot/            # Plotting utilities
├── parsers/
│   └── NWchem_parser.py
├── requirements.txt
├── run.sh
└── README.md
```

---

## What It Does

- Parses NWChem DFT frequency calculation output files (`.out`)
- Extracts total DFT energy, geometry convergence status, vibrational frequencies (cm⁻¹), and IR intensities (KM/mol)
- Bins IR intensities into a fixed-width wavenumber grid (50–4000 cm⁻¹, step 5) suitable for ML model input
- Incrementally updates the output CSV — already processed files are skipped on subsequent runs
- Supports multiple basis set folders, tracking files by relative path to avoid duplicate processing

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

**Step 0** — Run your NWChem frequency calculations. Each calculation must include both `task dft optimize` and `task dft freq` to produce the IR data needed by this pipeline.

**Step 1** — Place your `.out` files inside the `data/` folder. You can organize them into subfolders by basis set:
```
data/
├── 6-31G/
│   └── glycine_nwchem.out
└── cc-pVDZ/
    └── glycine_nwchem.out
```

**Step 2** — Run the parser:
```bash
python parsers/NWchem_parser.py
```


**Output** — Results are saved to `data/nwchem_results.csv`. Each row is one molecule with columns for energy, convergence, and IR intensity at each wavenumber bin (`spec_50`, `spec_55`, ..., `spec_4000`).

---


## Output Format

| filename | energy | converged | spec_50 | spec_55 | ... | spec_4000 |
|---|---|---|---|---|---|---|
| data/6-31G/glycine.out | -284.42 | True | 0.0 | 0.0 | ... | 1.785 |

---

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn
