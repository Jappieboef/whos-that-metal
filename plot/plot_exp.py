import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('/Users/jasper_laptop/Documents/REU/IR_SPEC_pipeline/data/nwchem_results.csv')
def gaussian_broadening(x, x0, sigma, intensity):
    return intensity * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

for index, row in df.iterrows():
    filename = row['filename']
    energy = row['energy']
    
    freq_cols = [col for col in df.columns if col.startswith('freq_')]
    int_cols = [col for col in df.columns if col.startswith('intensity_')]
    
    freqs = row[freq_cols].values.astype(float)
    intensities = row[int_cols].values.astype(float)
    
    valid_mask = ~np.isnan(freqs) & ~np.isnan(intensities)
    freqs = freqs[valid_mask]
    intensities = intensities[valid_mask]
    
    if len(freqs) == 0:
        print(f"Skipping {filename}: No vibrational data found.")
        continue
        
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.vlines(freqs, 0, intensities, color='crimson', linewidth=1.5, label='Stick Spectrum')
    
    x_range = np.linspace(min(freqs) - 100, max(freqs) + 100, 1000)
    y_range = np.zeros_like(x_range)
    sigma = 15.0  # Adjust this value to change peak broadening width
    
    for f, i in zip(freqs, intensities):
        y_range += gaussian_broadening(x_range, f, sigma, i)
        
    ax.plot(x_range, y_range, color='royalblue', alpha=0.8, label=f'Broadened Curve ($\sigma={sigma}$)')
    
    ax.set_xlabel('Frequency ($cm^{-1}$)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title(f'Vibrational Spectrum: {filename}\nEnergy = {energy:.4f} a.u. (Converged: {row["converged"]})', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    output_filename = f"{filename.split('.')[0]}_spectrum.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    print(f"Successfully generated and saved plot: {output_filename}")