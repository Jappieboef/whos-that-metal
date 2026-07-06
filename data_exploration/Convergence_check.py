import pandas as pd

df = pd.read_csv("data/xtb_results_cleantest.csv")
print(df.info()) # Note: Added () to properly call info()

conv_cnt = df['converged'].sum()

total_rows = len(df)

perc = (conv_cnt / total_rows) * 100 if total_rows > 0 else 0

print(f"{perc:.2f}% was converged")
