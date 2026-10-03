# Obtaining the Wheat flour data

## Redistribution notice

The processed `data_train.csv` and `data_test.csv` files are included in NIRBENCH-DL under written redistribution permission from Jean-Michel Roger / ChemHouse. See the repository [permission record](../../CHEMHOUSE_PERMISSION.md) for the redistribution statement and covered tasks. No standard dataset license was specified; the software Apache License 2.0 does not apply to these data. Preserve the original creator attribution, source links, and descriptions of modifications.

Downloading and preparing the original files remains an optional reproducibility route; it is not required when the distributed CSV files are present.

## Original source

- ChemProject ChemData page: <https://www.chemproject.org/chemdata>
- Spectra: <https://www.chemproject.org/media/data/x_140farines>
- References: <https://www.chemproject.org/media/data/y_140farines>
- Original attribution: D. Bertrand / INRA

## Benchmark preparation

Join the spectra and reference tables by row, retain all 525 spectral variables and the `PROTREF` protein target, remove non-model metadata, and place the target in the final column. The fixed 80/20 split uses `random_state=123` and produces 112 training and 28 test rows.

To reconstruct the partitions, run `python data_split.py` from this directory, or run the adjacent `data_split.ipynb` interactively. Both workflows download the original source, reproduce the preparation described above, verify the result against any existing local CSVs, and write the benchmark files.

See the task entry in the repository root `DATA_LICENSES.md` for provenance and redistribution status.

The manifest hashes describe the distributed CSV files. A new export can have different byte hashes because of CSV formatting even when the numerical values and sample assignments are equivalent; see `datasets/MANIFEST.md`.
