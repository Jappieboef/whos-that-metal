import re
import os
import pandas as pd


def parse_nwchem_output(filepath:str): 
#keep a record of how freq were calculated (dft/mp2) 
    """
    Args:
        filepath (str): Takes the filepath of the nwchem output file.

    Returns:
        dict: A dictionary containing:
            - 'filename' (str): Base name of the file.
            - 'energy' (float or None): Total DFT energy in Hartrees.
            - 'converged' (bool): Whether geometry optimization converged.
            - 'frequencies' (list): Vibrational frequencies (cm⁻¹).
            - 'intensities' (list): IR intensities.

        Raises:
            FileNotFoundError: If the filepath does not exist.

        Example:
            result = parse_nwchem_output('glycine_nwchem.out')
            print(result['energy'])
    """

    results = {'filename': os.path.basename(filepath),'energy': None,'converged': False}
    frequencies = []
    intensities = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("File path does not exist")
    
    for line in lines:
        match = re.search(r'Total DFT energy\s*=\s*([-]?\d+\.\d+)', line)
        if match:
            results['energy'] = float(match.group(1))
        
        if 'Optimization converged' in line:
            results['converged'] = True

        freq_match = re.search(r'XXXXXXXXX', line) # still need to find the correct regex for this
        if freq_match:
            frequencies.append(float(freq_match.group(1)))
        
        int_match = re.search(r'XXXXXXXXXXX', line) # still need to find the correct regex for this
        if int_match:
            intensities.append(float(int_match.group(1)))
        
    
    #adds an additional key-value pair for each frequency and intensity
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