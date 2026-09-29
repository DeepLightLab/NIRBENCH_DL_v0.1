# Obtaining the Wheat flour data

## Redistribution notice

For legal and licensing reasons, NIRBENCH-DL does not claim permission to redistribute the processed wheat-flour data while written permission is pending. Obtain the data from the original provider and generate `data_train.csv` and `data_test.csv` locally. Do not commit redistributed copies unless the rights holder grants permission.

## Original source

- ChemProject ChemData page: <https://www.chemproject.org/chemdata>
- Spectra: <https://www.chemproject.org/media/data/x_140farines>
- References: <https://www.chemproject.org/media/data/y_140farines>
- Original attribution: D. Bertrand / INRA

## Benchmark preparation

Join the spectra and reference tables by row, retain all 525 spectral variables and the `PROTREF` protein target, remove non-model metadata, and place the target in the final column. The fixed 80/20 split uses `random_state=123` and produces 112 training and 28 test rows.

Run `python data_split.py` from this directory, or run the adjacent `data_split.ipynb` interactively. Both workflows download the original source, reproduce the preparation described above, verify the result against any existing local CSVs, and write the benchmark files.

See the task entry in the repository root `DATA_LICENSES.md` for provenance and redistribution status.
