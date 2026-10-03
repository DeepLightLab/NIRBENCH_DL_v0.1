# Obtaining the Olive oil data

## Redistribution notice

The processed `data_train.csv` and `data_test.csv` files are included in NIRBENCH-DL under written redistribution permission from Jean-Michel Roger / ChemHouse. See the repository [permission record](../../CHEMHOUSE_PERMISSION.md) for the redistribution statement and covered tasks. No standard dataset license was specified; the software Apache License 2.0 does not apply to these data. Preserve the original creator attribution, source links, and descriptions of modifications.

Downloading and preparing the original files remains an optional reproducibility route; it is not required when the distributed CSV files are present.

## Original source

- ChemProject ChemData page: <https://www.chemproject.org/chemdata>
- NIR spectra: <https://www.chemproject.org/media/data/pir>
- Fatty-acid analyses: <https://www.chemproject.org/media/data/ags>
- Original attribution: N. Dupuy group / Aix-Marseille University

## Benchmark preparation

Match `pir.csv` and `ags.csv` by row, retain the 612 NIR variables, select palmitic acid `C16:0` as the target, and place the target in the final column. The fixed 80/20 split uses `random_state=123` and produces 149 training and 38 test rows.

To reconstruct the partitions, run `python data_split.py` from this directory, or run the adjacent `data_split.ipynb` interactively. Both workflows download the original source, reproduce the preparation described above, verify the result against any existing local CSVs, and write the benchmark files.

See the task entry in the repository root `DATA_LICENSES.md` for provenance and redistribution status.

The manifest hashes describe the distributed CSV files. A new export can have different byte hashes because of CSV formatting even when the numerical values and sample assignments are equivalent; see `datasets/MANIFEST.md`.
