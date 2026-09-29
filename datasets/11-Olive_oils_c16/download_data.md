# Obtaining the Olive oil data

## Redistribution notice

For legal and licensing reasons, NIRBENCH-DL does not claim permission to redistribute the processed olive-oil data while written permission is pending. Obtain the data from the original provider and generate `data_train.csv` and `data_test.csv` locally. Do not commit redistributed copies unless the rights holder grants permission.

## Original source

- ChemProject ChemData page: <https://www.chemproject.org/chemdata>
- NIR spectra: <https://www.chemproject.org/media/data/pir>
- Fatty-acid analyses: <https://www.chemproject.org/media/data/ags>
- Original attribution: N. Dupuy group / Aix-Marseille University

## Benchmark preparation

Match `pir.csv` and `ags.csv` by row, retain the 612 NIR variables, select palmitic acid `C16:0` as the target, and place the target in the final column. The fixed 80/20 split uses `random_state=123` and produces 149 training and 38 test rows.

Run `python data_split.py` from this directory, or run the adjacent `data_split.ipynb` interactively. Both workflows download the original source, reproduce the preparation described above, verify the result against any existing local CSVs, and write the benchmark files.

See the task entry in the repository root `DATA_LICENSES.md` for provenance and redistribution status.
