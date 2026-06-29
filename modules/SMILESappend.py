import os
import pandas as pd


#Loads the nwchem results and the testresults to merge them together. 
nwchem= pd.read_csv('data/nwchem_results.csv')
testfile= pd.read_csv('data/fake_test_results.csv', header=None)


#this keeps track of only the relevant columns and renames them to match the nwchem results. It also removes any whitespace from the smiles column.
testfile= testfile.iloc[:, [0, 1, 6]]
testfile.columns = ["amino_acid","metal","smiles"]
testfile["smiles"] = testfile["smiles"].str.strip()

#this merges the two dataframes together on the amino_acid and metal columns, and fills in any missing smiles values in the nwchem dataframe with the corresponding values from the testfile dataframe.
merged = nwchem[["amino_acid","metal"]].merge(
    testfile[["amino_acid","metal","smiles"]], on=["amino_acid","metal"], how="left"
)

#Fills missing SMILES in nwchem with values from testfile.
nwchem["smiles"] = nwchem["smiles"].fillna(merged["smiles"])

#this overwrites the original nwchem_results.csv file with the updated dataframe that now includes the SMILES values.
nwchem.to_csv(os.path.join('data/nwchem_results.csv'), index=False)

#print to verify that the SMILES values have been successfully appended to the nwchem_results.csv file and that code was successfully executed.
print("SMILES appended to nwchem_results.csv")