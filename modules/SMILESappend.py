import os
import pandas as pd

nwchem= pd.read_csv('data/nwchem_results.csv')
testfile= pd.read_csv('data/fake_test_results.csv', header=None)

print(nwchem.columns.tolist())
print(testfile.columns.tolist())


testfile= testfile.iloc[:, [0, 1, 6]]
testfile.columns = ["amino_acid","metal","smiles"]
testfile["smiles"] = testfile["smiles"].str.strip()

merged = nwchem[["amino_acid","metal"]].merge(
    testfile[["amino_acid","metal","smiles"]], on=["amino_acid","metal"], how="left"
)
nwchem["smiles"] = nwchem["smiles"].fillna(merged["smiles"])

nwchem.to_csv(os.path.join('data/nwchem_results.csv'), index=False)

