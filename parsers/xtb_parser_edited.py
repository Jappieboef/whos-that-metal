import re
import os
import pandas as pd
import numpy as np
import argparse

def bin_peaks_to_dict(frequencies: list, intensities: list, start=50, end=4000, step=5):
    # NOTE: if you change the step it cannot be parsed into the same dataframe as previous data — must be homogeneous
    """Bin peaks into nearest wavenumber bin, no broadening."""
    spectrum_dict = {int(wn): 0.0 for wn in np.arange(start, end + step, step)}
 
    for freq, intensity in zip(frequencies, intensities):
        nearest_bin = int(round(freq / step) * step)
        nearest_bin = max(start, min(end, nearest_bin))
        spectrum_dict[nearest_bin] += intensity
 
    return spectrum_dict


def parse_xtb_output(filepath: str):
    """
    Parse an XTB output file for energy, convergence, IR frequencies, and intensities.
 
    Args:
        filepath (str): Path to the XTB output file.
    """
    results = {
        'filename':     os.path.basename(filepath),
        'amino_acid':   None,
        'metal':        None,
        'mol_name':     None,
        'confirmation': None,
        'basis':        None,
        'theory':       None,
        'smiles':       np.nan,
        'energy':       None,
        'converged':    False,
    }

    un_parsed_name = os.path.splitext(results['filename'])[0]
    un_parsed_name_lst = un_parsed_name.split('_')

    results['amino_acid']   = un_parsed_name_lst[0]
    results['metal']        = un_parsed_name_lst[1]
    results['mol_name']     = un_parsed_name_lst[0] + "_" + un_parsed_name_lst[1]
    results['confirmation'] = un_parsed_name_lst[2]
    results['basis']        = un_parsed_name_lst[3]
    results['theory']       = un_parsed_name_lst[4]

    frequencies = []
    intensities = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return results

    eigval_pattern = re.compile(r'^\s*eigval\s*:\s*(.*)')
    inten_header_pattern = re.compile(r'^\s*IR intensities')
    inten_value_pattern = re.compile(r'\d+:\s*(-?[\d.]+)')
    energy_pattern = re.compile(r'^\s*\|\s*TOTAL ENERGY\s+([-\d.]+)\s+Eh')

    in_inten_block = False
    in_eigval_block = False

    for line in lines:
        #total energy
        energy_match = energy_pattern.match(line)
        if energy_match:
            results['energy'] = float(energy_match.group(1))
            continue

        if 'GEOMETRY OPTIMIZATION CONVERGED' in line:
            results['converged'] = True
            continue

        # --- frequencies ---
        eigval_match = eigval_pattern.match(line)
        if eigval_match:
            if not in_eigval_block:
                # xtb reprints this section verbatim earlier in the file - only the
                # last (final) block is the one that pairs up with IR intensities below.
                frequencies = []
            frequencies.extend(float(v) for v in eigval_match.group(1).split())
            in_eigval_block = True
            continue
        in_eigval_block = False

        # --- IR intensities ---
        if inten_header_pattern.match(line):
            in_inten_block = True
            continue

        if in_inten_block:
            if 'intensities' in line.lower() and 'IR' not in line:
                in_inten_block = False
                continue
            values = inten_value_pattern.findall(line)
            if values:
                intensities.extend(float(v) for v in values)
            else:
                in_inten_block = False

    spectrum = bin_peaks_to_dict(frequencies, intensities, step=5)

    results.update({f'spec_{wn}': val for wn, val in spectrum.items()})

    return results


def process_folder(folder_path: str, legacy_df: pd.DataFrame):
    """
    Walk a folder and parse every .out file that hasn't been processed yet.
 
    Args:
        folder_path (str): Directory containing xTB .out files.
        legacy_df (pd.DataFrame): Existing results CSV loaded as a
            DataFrame, or None on the first run.
 
    Returns:
        list: Parsed result dictionaries ready to be turned into a DataFrame.
    """
    already_processed = set()
    if legacy_df is not None:
        already_processed = set(legacy_df['filename'].tolist())

    all_results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.out'):
            filepath = os.path.relpath(os.path.join(folder_path, filename))
            if filename in already_processed:
                print(f"Skipping {filepath} — already in CSV")
                continue

            base = os.path.splitext(filename)[0]
            if len(base.split('_')) < 5:
                print(f"Skipping {filepath} — filename doesn't match expected naming pattern")
                continue

            results = parse_xtb_output(filepath)
            all_results.append(results)

    return all_results


def data_frame_results(all_results):
    """
    Returns:
        dataframe : Pandas Dataframe from the dictionary.
    """
    df = pd.DataFrame(all_results)
    return df


def data_frame_concat(df, legacy_df):
    """
    Args:
        df (DF): The newly parsed data dataframe
        legacy_df (DF): brings in the already existing dataframe

    Returns:
        dataframe : A concatinated version of the two dataframes
    """
    combined_df = pd.concat([legacy_df, df], ignore_index=True)
    return combined_df


def main():
    parser = argparse.ArgumentParser(description='Parse xTB output files')
    parser.add_argument('--folder', type=str, default='/work/simcodes/IRSpectra/XTB/best_conformers/metal_dimers',
                        help='Folder containing .out files')
    parser.add_argument('--output', type=str, default='/work/simcodes/IRSpectra/Parsed_Data/xtb_results.csv',
                        help='Path to output CSV')
    args = parser.parse_args()

    folder_path = args.folder
    output_path = args.output

    if os.path.exists(output_path):
        legacy_df = pd.read_csv(output_path)
    else:
        legacy_df = None

    all_results = process_folder(folder_path, legacy_df)

    df = data_frame_results(all_results)
    if legacy_df is not None:
        combined_df = data_frame_concat(df, legacy_df)
    else:
        combined_df = df

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving to: {output_path}")
    print(f"Rows: {len(combined_df)}")
    combined_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()