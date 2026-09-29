# Obtaining the IDRC pharmaceutical-tablet data

## Redistribution notice

For legal and licensing reasons, NIRBENCH-DL does not claim permission to redistribute the processed IDRC 2002 Shootout data while written permission is pending. Obtain the data from the original provider and generate `data_train.csv` and `data_test.csv` locally. Do not commit redistributed copies unless the rights holder grants permission.

## Original source

- Eigenvector dataset page: <https://eigenvector.com/resources/data-sets/>
- Official Eigenvector archive: <https://eigenvector.com/wp-content/uploads/2019/06/nir_shootout_2002.mat_.zip>
- Original source: 2002 International Diffuse Reflectance Conference Shootout

## Benchmark preparation

Select spectrometer 1, its 650 spectral variables, and the active-ingredient assay target. Combine the source calibration and source test observations into the 615-row benchmark training partition, retain the 40-row source validation partition as the benchmark test partition, round `assay` to its reported one-decimal precision, convert the arrays to CSV, and place `assay` in the final column.

Run `python data_split.py` from this directory, or run the adjacent `data_split.ipynb` interactively. Both workflows download the original source, reproduce the preparation described above, verify the result against any existing local CSVs, and write the benchmark files.

See the task entry in the repository root `DATA_LICENSES.md` for provenance and redistribution status.
