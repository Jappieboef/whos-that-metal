
import re
import os
import pandas as pd
def parse_nwchem_output(filepath):
    results = {
        'filename': os.path.basename(filepath),
        'energy': None,
        'converged': False,
    }
    
    frequencies = []
    intensities = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        match = re.search(r'Total DFT energy\s*=\s*([-]?\d+\.\d+)', line)
        if match:
            results['energy'] = float(match.group(1))
        
        if 'Optimization converged' in line:
            results['converged'] = True
        
    for i, freq in enumerate(frequencies):
        results[f'freq_{i+1}'] = freq
    
    for i, intensity in enumerate(intensities):
        results[f'intensity_{i+1}'] = intensity
    
    return results

def process_folder(folder_path):
    all_results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.out'):
            filepath = os.path.join(folder_path, filename)
            results = parse_nwchem_output(filepath)
            all_results.append(results)
    return all_results

def data_frame_results(all_results):
    df = pd.DataFrame(all_results)
    return df

def main():
    folder_path = '/Users/jasper_laptop/Documents/REU/IR_SPEC_pipeline/data'
    all_results = process_folder(folder_path)
    for results in all_results:
        print(results)
    df = data_frame_results(all_results)
    print(df)
    
if __name__ == "__main__":
    main()