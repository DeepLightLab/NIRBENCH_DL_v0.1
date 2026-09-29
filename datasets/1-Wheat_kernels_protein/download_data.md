# Obtaining the Wheat kernels data

## Redistribution notice

For legal and licensing reasons, NIRBENCH-DL does not claim permission to redistribute the processed Wheat kernels data while written permission is pending. Obtain the data from the original provider and generate `data_train.csv` and `data_test.csv` locally. Do not commit redistributed copies unless the rights holder grants permission.

## Original source

- Dataset page: <https://ucphchemometrics.com/datasets/>
- Official Wheat kernels archive: <https://sid.erda.dk/share_redirect/dLQ6VHNshw/Wheat%20kernels%20.zip>
- Associated paper: Nielsen et al. (2003), <https://doi.org/10.1094/CCHEM.2003.80.3.274>

## Benchmark preparation

The benchmark version extracts the NIR/NIT spectral matrix and protein reference from the MATLAB source, retains 100 spectral variables, converts the data to CSV, and places `protein` in the final column. Preserve the source `Calibration` partition as the 415-row benchmark training set and the source `Validation` partition as the 108-row benchmark test set.

Run `python data_split.py` from this directory, or run the adjacent `data_split.ipynb` interactively. Both workflows download the original source, reproduce the preparation described above, verify the result against any existing local CSVs, and write the benchmark files.

See the task entry in the repository root `DATA_LICENSES.md` for provenance and redistribution status.
