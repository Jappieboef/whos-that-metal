import os
import pandas as pd
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
    parent_folder = os.path.basename(os.path.dirname(filepath)) #finds the parent folder of the summary as it contains the key info of sample combination
    parts = parent_folder.split('_') #naming structure from input is constant AA(amind acid)_Cu2+(metal) and turns it into a list
    aa = parts[0] #we always know the first part is an Amino Acid 
    metal = parts[1] #second part is always the metal

    smiles = None
    coordination_number = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('msmiles'): #looking for the full SMILES
                    smiles = line.split('\t')[1].strip() #used the delminator of \t from the tsv format to parse
                if line.startswith('coordination_number'): #looking for the coordination numbers
                    coordination_number = line.split('\t')[1].strip()
    except FileNotFoundError: #just incase it catches an error with opening files
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
    ] #not enttirelt sure on this part ^ why do we need 3 combinations?
    

def process_directory(input_dir: str, legacy_df=None) -> list:
    already_processed = set()
    if legacy_df is not None:
        already_processed = set(
            zip(legacy_df['amino_acid'], legacy_df['metal'], legacy_df['conformation'])
        ) #makes a set that can easily be checked if that combination already resides within the CSV

    all_results = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == 'summary.tsv':
                filepath = os.path.join(root, file)
                results = parse_summary(filepath)
                for result in results:
                    key = (result['amino_acid'], result['metal'], result['conformation'])
                    if key in already_processed:
                        print(f"Skipping {key} — already in CSV")
                        continue #skips it if the data already exists in the CSV
                    all_results.append(result)

    return all_results


def main():
    """
    how to run demo - 
                    python parsers/Seq2Smile_parser.py --input AH_CO+2 --output data/smiles_results.csv
    """
    parser = argparse.ArgumentParser(description='Parse seq2mSMILES summary files')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to root directory containing summary.tsv files')
    parser.add_argument('--output', type=str, default='smiles_results.csv',
                        help='Path to output CSV file')
    args = parser.parse_args()


    #building the legacy dataframe same logic within NWchem_parser_V2 look to see full explained logic
    if os.path.exists(args.output):
        legacy_df = pd.read_csv(args.output)
    else:
        legacy_df = None

    all_results = process_directory(args.input, legacy_df)

    if all_results:
        df = pd.DataFrame(all_results)
        if legacy_df is not None: #the base case if no data has been established 
            combined_df = pd.concat([legacy_df, df], ignore_index=True)
        else:
            combined_df = df
        os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
        combined_df.to_csv(args.output, index=False)
        print(f"Added {len(all_results)} new records saved to {args.output}")
    else:
        print("No new records to process")


if __name__ == "__main__":
    main()

