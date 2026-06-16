import re
import os
import pandas as pd

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
                and 'Infra Red Intensities' in line
                and 'Projected' not in line
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
                frequencies.append(float(m.group(1)))      # frequency cm-1
                intensities.append(float(m.group(2)))      # KM/mol column
            elif frequencies and set(line.strip()) <= {'-'} and line.strip():
                in_freq_section = False
                already_parsed = True
 
    # Unpack into individual columns
    for i, freq in enumerate(frequencies):
        results[f'freq_{i+1}'] = freq
    for i, intensity in enumerate(intensities):
        results[f'intensity_{i+1}'] = intensity
 
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