import re
import os
import pandas as pd
import numpy as np
import argparse


def parse_summary(filepath: str) -> list:
    """
    Parse a single summary.tsv file and extract msmiles, coordination number.
    Returns 3 rows, one per conformation.

    Args:
        filepath (str): Path to the summary.tsv file

    Returns:
        list: list of 3 dicts containing AA, metal, smiles, coordination_number, conformation
    """
    parent_folder = os.path.basename(os.path.dirname(filepath))
    parts = parent_folder.split('_')
    aa = parts[0]
    metal = parts[1]

    smiles = None
    coordination_number = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('msmiles'):
                    smiles = line.split('\t')[1].strip()
                if line.startswith('coordination_number'):
                    coordination_number = line.split('\t')[1].strip()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []

    return [
        {
            'amino_acid': aa,
            'metal': metal,
            'smiles': smiles,
            'coordination_number': coordination_number,
            'conformation': conf
        }
        for conf in [1, 2, 3]
    ]
    

def process_directory(input_dir: str) -> list:
    all_results = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == 'summary.tsv':
                filepath = os.path.join(root, file)
                results = parse_summary(filepath)
                all_results.extend(results)

    return all_results

def main():
    parser = argparse.ArgumentParser(description='Parse seq2mSMILES summary files')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to root directory containing summary.tsv files')
    parser.add_argument('--output', type=str, default='smiles_results.csv',
                        help='Path to output CSV file')
    args = parser.parse_args()

    all_results = process_directory(args.input)
    
    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    df.to_csv(args.output, index=False)
    print(f"Parsed {len(all_results)} records saved to {args.output}")


if __name__ == "__main__":
    main()