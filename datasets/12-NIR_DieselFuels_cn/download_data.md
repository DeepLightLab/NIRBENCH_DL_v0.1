# Obtaining the SWRI diesel data

## Redistribution notice

For legal and licensing reasons, NIRBENCH-DL does not claim permission to redistribute the processed diesel-fuel data while written permission is pending. Obtain the data from the original provider and generate `data_train.csv` and `data_test.csv` locally. Do not commit redistributed copies unless the rights holder grants permission.

## Original source

- Eigenvector dataset page: <https://eigenvector.com/resources/data-sets/>
- Official CSV archive: <https://eigenvector.com/wp-content/uploads/2019/06/SWRI_Diesel_NIR_CSV.zip>
- Original source: Southwest Research Institute; the source page notes U.S. Army project sponsorship

## Benchmark preparation

Align `diesel_spec.csv` and `diesel_prop.csv` by row, select cetane number (`CN`), split all 784 rows 80/20 with `random_state=123`, and then remove rows lacking `CN` separately from each partition. Retain the 401 spectral variables and place `CN` in the final column; this produces 298 training and 83 test rows.

Run `python data_split.py` from this directory, or run the adjacent `data_split.ipynb` interactively. Both workflows download the original source, reproduce the preparation described above, verify the result against any existing local CSVs, and write the benchmark files.

See the task entry in the repository root `DATA_LICENSES.md` for provenance and redistribution status.
