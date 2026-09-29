# Obtaining the CGL NIR data

## Redistribution notice

For legal and licensing reasons, NIRBENCH-DL does not claim permission to redistribute the processed CGL data while written permission is pending. Obtain the data from the original provider and generate `data_train.csv` and `data_test.csv` locally. Do not commit redistributed copies unless the rights holder grants permission.

## Original source

- Eigenvector dataset page: <https://eigenvector.com/resources/data-sets/>
- Official archive: <https://eigenvector.com/wp-content/uploads/2021/04/CGL_nir.mat_.zip>
- Original contributors: Tormod Næs and Tomas Isaksson

## Benchmark preparation

Load `Xcal`, `Xtest`, `Ycal`, and `Ytest` from `CGL_nir.mat`. Retain the 117 spectral variables, select `Glucose (wt %)` from the response arrays, preserve the supplied 153-row calibration and 78-row test division, convert both partitions to CSV, and place the glucose target in the final column.

Despite the historical benchmark folder name, the detailed source description identifies CGL as a casein/glucose/lactate mixture-design dataset rather than a grain dataset.

Run `python data_split.py` from this directory, or run the adjacent `data_split.ipynb` interactively. Both workflows download the original source, reproduce the preparation described above, verify the result against any existing local CSVs, and write the benchmark files.

See the task entry in the repository root `DATA_LICENSES.md` for provenance and redistribution status.
