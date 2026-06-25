import re
import os
import pandas as pd
import numpy as np
import argparse


def bin_peaks_to_dict(frequencies:list, intensities:list, start=50, end=4000, step=5): #change the step for fine tuning
    #NOTE if you change the step it cannot be parsed into the same dataframe as pervious data MUST be Homogenus
    """Bin peaks into nearest wavenumber bin, no broadening."""
    spectrum_dict = {int(wn): 0.0 for wn in np.arange(start, end + step, step)}
    
    for freq, intensity in zip(frequencies, intensities): #zips the contents of feq and int together so they can be itterated together to esure parrell connections
        nearest_bin = int(round(freq / step) * step) #rounding math to put it in its closest bin
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
    #initialize the dictonary that will be returned at the end of the function, later to be converted to a pd
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


    #every "sample" will need a new list of fequencies and intesnsities
    frequencies = []
    intensities = []
 
    try:                                                                    #try block possibly redundant <---
        with open(filepath, 'r') as f: #opens the specified "*.out" file
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return results # If a file is not found returns the results incase its a partial extraction
 
    #starts all the flags as False
    in_freq_section = False
    already_parsed = False
    dft_total_flag = False
 
    for line in lines: #itterated through every line of the .out file
        match_1 = re.search(r'NWChem Nuclear Hessian and Frequency Analysis', line) #sets off flag to ensure we grab the calculation we want
        if match_1: #check 1
            dft_total_flag = True
        if dft_total_flag:
            match_2 = re.search(r'Total DFT energy\s*=\s*([-]?\d+\.\d+)', line)
            if match_2: #check 2
                results['energy'] = float(match_2.group(1))
                dft_total_flag = False # turns the flag back off the moment it finds the total DFT energy specified
 
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
                r'[-]?\d+\.\d+\s+[-]?\d+\.\d+\s+([-]?\d+\.\d+)\s+[-]?\d+\.\d+', #long line of regular expression built with https://regex101.com/
                line
            )
            if match:
                frequencies.append(float(match.group(1)))      # frequency cm-1
                intensities.append(float(match.group(2)))      # KM/mol column
            elif frequencies and set(line.strip()) <= {'-'} and line.strip(): #looking for the end of the specific freq we want
                in_freq_section = False
                already_parsed = True #setting the flags to ensure only the data we want is collected and nothing more
 
    spectrum = bin_peaks_to_dict(frequencies, intensities, step=5) #calls the binning function from above
    results.update({f'spec_{wn}': val for wn, val in spectrum.items()}) # appends the newly binned data to the results dict
 
    return results

def process_folder(folder_path:str,legacy_df:pd.DataFrame):
    """
    Args:
        folder_path (str): takes in the folder path to where all the data resides.
        legacy_df (pd): takes in either an exisiting dataframe from previous runs or is set to none for the first itteration

    Returns:
        list : returns a list of dictionaries containing the parsed data from 'parse_nwchem_out'
    """

    already_processed = set()
    if legacy_df is not None: #meaning a legacy DF exists now have to put it in a set
        already_processed = set(legacy_df['filename'].tolist())
    
    all_results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.out'):
            filepath = os.path.relpath(os.path.join(folder_path, filename))
            if filename in already_processed:
                print(f"Skipping {filepath} — already in CSV")
                continue #skips the itteration if the filepath already exsits meaning it will not be reparsed
            results = parse_nwchem_output(filepath)
            all_results.append(results)
    
    return all_results

def data_frame_results(all_results):
    """
    Returns:
        dataframe : Pandas Dataframe from the dictionary produced from the 'parse_nwchem_output'
    """

    df = pd.DataFrame(all_results) #calls from the pandas library
    return df

def data_frame_concat(df,legacy_df):
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
    parser = argparse.ArgumentParser(description='Parse NWChem output files')
    parser.add_argument('--folder', type=str, default='data',
                        help='Folder containing .out files')
    parser.add_argument('--output', type=str, default='data/nwchem_results.csv',
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
    combined_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()