import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def gaussian_broaden_spectra(df: pd.DataFrame, step: int = 5, fwhm: float = 20.0, spec_prefix: str = "spec_") -> pd.DataFrame:
    """Convolve each row's binned spectrum with a Gaussian of the given FWHM (cm-1).

    The parser bins peaks into their nearest wavenumber bin with no broadening,
    so adjacent spec_ columns are otherwise uncorrelated - this smooths them so
    linear compression (SVD/PCA) has actual structure to exploit.
    """
    spec_cols = sorted(
        (c for c in df.columns if c.startswith(spec_prefix)),
        key=lambda c: int(c.split("_")[1]),
    )

    sigma_bins = fwhm / (2 * np.sqrt(2 * np.log(2))) / step
    intensities = df[spec_cols].to_numpy(dtype=float)
    broadened = gaussian_filter1d(intensities, sigma=sigma_bins, axis=1)

    out = df.copy()
    out[spec_cols] = broadened
    return out


def main():
    """
    how to run demo -
                    python modules/broaden_spectrum.py --input data/nwchem_results.csv --output data/nwchem_results_broadened.csv
    """
    parser = argparse.ArgumentParser(description="Apply Gaussian broadening to binned NWChem IR spectra")
    parser.add_argument("--input", type=str, default="data/nwchem_results.csv",
                        help="Path to the parsed NWChem results CSV")
    parser.add_argument("--output", type=str, default="data/nwchem_results_broadened.csv",
                        help="Path to write the broadened CSV")
    parser.add_argument("--step", type=int, default=5, help="Wavenumber bin width used by the parser (cm-1)")
    parser.add_argument("--fwhm", type=float, default=20.0, help="Gaussian linewidth, full width at half max (cm-1)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    broadened = gaussian_broaden_spectra(df, step=args.step, fwhm=args.fwhm)
    broadened.to_csv(args.output, index=False)
    print(f"Broadened spectra saved to {args.output}")


if __name__ == "__main__":
    main()