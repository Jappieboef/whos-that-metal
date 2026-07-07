import re
import os
import pandas as pd
import numpy as np
import argparse


def bin_peaks_to_dict(frequencies, intensities, start=50, end=4000, step=5):
    """Bin peaks into nearest wavenumber bin, no broadening."""
    spectrum_dict = {int(wn): 0.0 for wn in np.arange(start, end + step, step)}
    for freq, intensity in zip(frequencies, intensities):
        nearest_bin = int(round(freq / step) * step)
        nearest_bin = max(start, min(end, nearest_bin))
        spectrum_dict[nearest_bin] += intensity
    return spectrum_dict


def parse_vibspectrum(filepath):
    """
    Parse an xTB `vibspectrum` file into the standard spectrum-row format.

    Metadata is taken from the PARENT DIRECTORY name, which follows
    AA_Metal_Conf_Basis_Theory (e.g. AA_Co+2_C10_0_xtb).

    Data rows look like:
        mode  symmetry   wavenumber   IR_intensity   selection
    Only modes flagged 'YES' (real vibrations, not translations/rotations)
    are kept.
    """
    parent = os.path.basename(os.path.dirname(filepath))

    results = {
        'dirname':      parent,
        'amino_acid':   None,
        'metal':        None,
        'mol_name':     None,
        'confirmation': None,
        'basis':        None,
        'theory':       None,
        'smiles':       np.nan,
        'converged':    True,   # a vibspectrum only exists if the freq job finished
    }

    parts = parent.split('_')
    if len(parts) >= 5:
        results['amino_acid']   = parts[0]
        results['metal']        = parts[1]
        results['mol_name']     = parts[0] + "_" + parts[1]
        results['confirmation'] = parts[2]
        results['basis']        = parts[3]
        results['theory']       = parts[4]

    frequencies = []
    intensities = []

    # mode  [symmetry]  wavenumber  intensity  selection
    # symmetry may be absent on the zero modes, so match flexibly and rely on YES
    row_pattern = re.compile(
        r'^\s*\d+\s+\S+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+YES\s*$'
    )

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or line.startswith('$'):
                    continue
                m = row_pattern.match(line)
                if m:
                    freq = float(m.group(1))
                    inten = float(m.group(2))
                    frequencies.append(freq)
                    intensities.append(inten)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return results

    if len(frequencies) != len(intensities):
        print(f"WARNING mismatch in {parent}: "
              f"{len(frequencies)} freq vs {len(intensities)} inten — skipping spectrum")
        results['freq_inten_mismatch'] = True
        return results

    results['freq_inten_mismatch'] = False
    spectrum = bin_peaks_to_dict(frequencies, intensities, step=5)
    results.update({f'spec_{wn}': val for wn, val in spectrum.items()})
    return results


def process_directory(input_dir, legacy_df):
    """Recursively find every `vibspectrum` file under input_dir and parse it."""
    already_processed = set()
    if legacy_df is not None and 'dirname' in legacy_df:
        already_processed = set(legacy_df['dirname'].tolist())

    all_results = []
    for root, dirs, files in os.walk(input_dir):
        if 'vibspectrum' in files:
            parent = os.path.basename(root)
            if parent in already_processed:
                print(f"Skipping {parent} — already in CSV")
                continue
            if len(parent.split('_')) < 5:
                print(f"Skipping {parent} — dir name doesn't match AA_Metal_Conf_Basis_Theory")
                continue
            filepath = os.path.join(root, 'vibspectrum')
            all_results.append(parse_vibspectrum(filepath))

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Parse xTB vibspectrum files')
    parser.add_argument('--folder', type=str,
                        default='/work/simcodes/IRSpectra/XTB/output_dir/metal_dimers',
                        help='Root folder containing per-molecule directories')
    parser.add_argument('--output', type=str,
                        default='/work/simcodes/IRSpectra/Parsed_Data/vibspectrum_results.csv',
                        help='Path to output CSV')
    args = parser.parse_args()


    legacy_df = pd.read_csv(args.output) if os.path.exists(args.output) else None
    all_results = process_directory(args.folder, legacy_df)

    if not all_results:
        print("No new vibspectrum files to process.")
        return

    df = pd.DataFrame(all_results)
    if legacy_df is not None:
        df = pd.concat([legacy_df, df], ignore_index=True)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
