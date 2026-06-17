import re
import os
import pandas as pd
import numpy as np

# def create_spectrum_dict(start=50, end=4000, step=5):
#     """Create zeroed spectrum dictionary with wavenumber keys."""
#     grid = np.arange(start, end + step, step)
#     return {int(wn): 0.0 for wn in grid}

def bin_peaks_to_dict(frequencies:list, intensities:list, start=50, end=4000, step=5):
    """Bin peaks into nearest wavenumber bin, no broadening."""
    spectrum_dict = {int(wn): 0.0 for wn in np.arange(start, end + step, step)}
 
    
    for freq, intensity in zip(frequencies, intensities):
        nearest_bin = int(round(freq / step) * step)
        nearest_bin = max(start, min(end, nearest_bin))
        spectrum_dict[nearest_bin] += intensity
    
    return spectrum_dict




def parse_nwchem_output(filepath: str):
    """
    Parse a NWChem output file for energy, convergence, IR frequencies and intensities.
 
    Args:
        filepath (str): Path to the nwchem output file.
 
    Returns:
        dict: filename, energy, converged, and freq_i / intensity_i columns.
    """
    results = {
        'filename': os.path.basename(filepath),
        'energy': None,
        'converged': False,
        
    }
    frequencies = []
    intensities = []
 
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return results
 
    in_freq_section = False
    already_parsed = False
 
    for line in lines:
        match = re.search(r'Total DFT energy\s*=\s*([-]?\d+\.\d+)', line)
        if match:
            results['energy'] = float(match.group(1))
 
        if 'Optimization converged' in line:
            results['converged'] = True
 
        if ('Normal Eigenvalue' in line
                and 'Infra Red Intensities' in line # might need to changed to projected
                and 'Projected' in line
                and not already_parsed):
            in_freq_section = True
            continue  # header line itself has no data
 
        if in_freq_section:
            match = re.search(
                r'^\s*\d+\s+([-]?\d+\.\d+)\s+\|\|\s+'
                r'[-]?\d+\.\d+\s+[-]?\d+\.\d+\s+([-]?\d+\.\d+)\s+[-]?\d+\.\d+',
                line
            )
            if match:
                frequencies.append(float(match.group(1)))      # frequency cm-1
                intensities.append(float(match.group(2)))      # KM/mol column
            elif frequencies and set(line.strip()) <= {'-'} and line.strip():
                in_freq_section = False
                already_parsed = True
 
    spectrum = bin_peaks_to_dict(frequencies, intensities, step=5)
    results.update({f'spec_{wn}': val for wn, val in spectrum.items()})
 
    return results

def process_folder(folder_path):
    """
    Args:
        folder_path (str): takes in the folder path to where all the data resides.

    Returns:
        list : returns a list of dictionaries containing the parsed data from 'parse_nwchem_out'
    """

    all_results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.out'):
            filepath = os.path.join(folder_path, filename)
            results = parse_nwchem_output(filepath)
            all_results.append(results)
    return all_results

def data_frame_results(all_results):
    """
    Returns:
        dataframe : Pandas Dataframe from the dictionary produced from the 'parse_nwchem_output'
    """

    df = pd.DataFrame(all_results)
    return df

def main():
    folder_path = 'data'
    output_path = 'data/nwchem_results.csv' 
    #how do I make this better? does it need to be improved?^
    
    all_results = process_folder(folder_path)
    for results in all_results:
        print(results)
    
    df = data_frame_results(all_results)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()